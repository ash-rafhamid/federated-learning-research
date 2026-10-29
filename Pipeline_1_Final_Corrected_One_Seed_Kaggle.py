# %% [markdown]
# # Pipeline 1: corrected federated-learning experiment
# Run this notebook once per seed. Change only `RUN_SEED` in the configuration cell.
# Private images remain inside conceptual client objects; the server receives model
# messages and aggregate metrics only.

# %%
# Install Opacus when Kaggle does not already provide it.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("opacus") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opacus", "-q"])

# %%
# Imports
import copy
import io
import json
import math
import random
import shutil
import struct
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from torch.utils.data import DataLoader, TensorDataset

# %% [markdown]
# ## 1. Experiment configuration

# %%
# Change only this value in the three Kaggle sessions: 42, 123, and 2026.
RUN_SEED = 42

NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 15
ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
DIRICHLET_ALPHA = 0.5
MIN_CLIENT_SIZE = 128
VALIDATION_FRACTION = 0.10
EVALUATE_EVERY = 5

PUBLIC_CLASSES = 36
PRIVATE_CLASSES = 47
PUBLIC_SAMPLE = 60_000
PUBLIC_PRETRAIN_EPOCHS = 5
PUBLIC_PRETRAIN_BATCH = 128

MAX_GRAD_NORM = 1.0
DELTA = 1e-5
PRIVACY_SETTINGS = {
    "NoDP": None,
    "sigma0.5": 0.5,
    "sigma1.0": 1.0,
    "sigma2.0": 2.0,
}

# LR is selected with validation data, never with test data.
LR_CANDIDATES = [0.3, 0.5]
LR_TUNING_ROUNDS = 5

# All methods use 32-bit. The proposed frozen method also uses 8/4-bit.
METHODS = ["baseline", "pretrained", "frozen"]
BITS_BY_METHOD = {
    "baseline": [32],
    "pretrained": [32],
    "frozen": [32, 8, 4],
}

PUBLIC_CSV = Path(
    "/kaggle/input/datasets/swarupshah/digit-character/"
    "digit_char_splited_dataset.csv"
)
EMNIST_TRAIN_CSV = Path(
    "/kaggle/input/datasets/crawford/emnist/emnist-balanced-train.csv"
)
EMNIST_TEST_CSV = Path(
    "/kaggle/input/datasets/crawford/emnist/emnist-balanced-test.csv"
)

OUTPUT_DIR = Path(f"/kaggle/working/pipeline1_final_seed_{RUN_SEED}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

RESULTS_JSON = OUTPUT_DIR / f"results_seed_{RUN_SEED}.json"
BODY_CHECKPOINT = OUTPUT_DIR / f"public_body_seed_{RUN_SEED}.pt"
SELECTED_LR_JSON = OUTPUT_DIR / f"selected_learning_rates_seed_{RUN_SEED}.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Seed: {RUN_SEED} | Device: {DEVICE}")
print(f"Clients per round: {CLIENTS_PER_ROUND}/{NUM_CLIENTS}")

# %% [markdown]
# ## 2. Reproducibility and safe saving

# %%
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(value), handle, indent=2)
    temporary.replace(path)


seed_everything(RUN_SEED)

# %% [markdown]
# ## 3. Load and partition the datasets

# %%
def load_public(path: Path, sample_count: int, seed: int):
    frame = pd.read_csv(path)
    lower_names = [str(column).lower() for column in frame.columns]
    label_column = (
        frame.columns[lower_names.index("label")]
        if "label" in lower_names
        else frame.columns[-1]
    )
    labels = frame[label_column].to_numpy(dtype=np.int64)
    pixels = frame.drop(columns=[label_column]).to_numpy(dtype=np.float32)[:, :784]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(labels), min(sample_count, len(labels)), replace=False)
    images = pixels[chosen].reshape(-1, 1, 28, 28) / 255.0
    sampled_labels = labels[chosen]
    if sampled_labels.min() < 0 or sampled_labels.max() >= PUBLIC_CLASSES:
        raise ValueError("Public labels must be in the range 0..35.")
    return torch.from_numpy(images), torch.from_numpy(sampled_labels)


def load_emnist(path: Path):
    frame = pd.read_csv(path, header=None)
    labels = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    images = frame.iloc[:, 1:785].to_numpy(dtype=np.float32).reshape(-1, 28, 28)
    # EMNIST CSV images require this orientation correction.
    images = np.transpose(images, (0, 2, 1))[:, None, :, :] / 255.0
    if labels.min() < 0 or labels.max() >= PRIVATE_CLASSES:
        raise ValueError("EMNIST Balanced labels must be in the range 0..46.")
    return torch.from_numpy(images), torch.from_numpy(labels)


def dirichlet_partition(labels, number_of_clients, alpha, minimum_size, seed):
    label_array = labels.numpy()
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        partitions = [[] for _ in range(number_of_clients)]
        for class_id in range(PRIVATE_CLASSES):
            class_indices = np.where(label_array == class_id)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet([alpha] * number_of_clients)
            cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            for client_id, client_indices in enumerate(np.split(class_indices, cuts)):
                partitions[client_id].extend(client_indices.tolist())
        arrays = [np.asarray(items, dtype=np.int64) for items in partitions]
        if min(map(len, arrays)) >= minimum_size:
            combined = np.concatenate(arrays)
            if len(np.unique(combined)) != len(labels):
                raise RuntimeError("Training partition duplicated or lost examples.")
            return arrays
    raise RuntimeError("Could not create the requested non-IID client partition.")


def balanced_test_partition(labels, number_of_clients, seed):
    # All official test examples are assigned once to evaluation clients.
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(number_of_clients)]
    label_array = labels.numpy()
    for class_id in range(PRIVATE_CLASSES):
        class_indices = np.where(label_array == class_id)[0]
        rng.shuffle(class_indices)
        for offset, index in enumerate(class_indices):
            client_id = (offset + class_id) % number_of_clients
            partitions[client_id].append(int(index))
    arrays = [np.asarray(items, dtype=np.int64) for items in partitions]
    combined = np.concatenate(arrays)
    if len(np.unique(combined)) != len(labels):
        raise RuntimeError("Test partition duplicated or lost examples.")
    return arrays


def split_local_train_validation(images, labels, fraction, seed):
    order = torch.randperm(len(labels), generator=make_generator(seed))
    validation_count = max(1, min(len(labels) - 1, round(len(labels) * fraction)))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    return (
        images[training_indices],
        labels[training_indices],
        images[validation_indices],
        labels[validation_indices],
    )


def build_schedule(rounds, number_of_clients, clients_per_round, seed):
    rng = np.random.default_rng(seed)
    return [
        rng.choice(number_of_clients, clients_per_round, replace=False).tolist()
        for _ in range(rounds)
    ]

# %% [markdown]
# ## 4. CNN and public pre-training

# %%
class CNN(nn.Module):
    FEATURE_DIMENSION = 128

    def __init__(self, number_of_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(self.FEATURE_DIMENSION, number_of_classes)

    def forward(self, images):
        return self.head(self.body(images))


def cpu_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def load_state(module, state):
    module.load_state_dict(state, strict=True)


@torch.no_grad()
def extract_with_body(body_state, images, batch_size=512):
    model = CNN(PRIVATE_CLASSES).to(DEVICE)
    model.body.load_state_dict(body_state)
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        outputs.append(model.body(images[start : start + batch_size].to(DEVICE)).cpu())
    return torch.cat(outputs)


def pretrain_public_body(images, labels):
    seed_everything(RUN_SEED + 1_000)
    model = CNN(PUBLIC_CLASSES).to(DEVICE)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=PUBLIC_PRETRAIN_BATCH,
        shuffle=True,
        generator=make_generator(RUN_SEED + 1_001),
        num_workers=0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    started = time.perf_counter()
    for epoch in range(PUBLIC_PRETRAIN_EPOCHS):
        model.train()
        running_correct = running_total = 0
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_correct += int((logits.argmax(1) == batch_labels).sum().item())
            running_total += len(batch_labels)
        print(
            f"Public epoch {epoch + 1}/{PUBLIC_PRETRAIN_EPOCHS}: "
            f"train accuracy={100 * running_correct / running_total:.2f}%"
        )
    body_state = cpu_state(model.body)
    # Normalization uses public features only.
    public_features = extract_with_body(body_state, images)
    public_mean = public_features.mean(0, keepdim=True)
    public_std = public_features.std(0, keepdim=True).clamp_min(1e-6)
    return body_state, public_mean, public_std, time.perf_counter() - started

# %% [markdown]
# ## 5. Real 32-bit, 8-bit, and packed 4-bit messages

# %%
SERIAL_MAGIC = b"P1S1"


def pack_four_bit(values: np.ndarray) -> bytes:
    unsigned = (values.astype(np.int16) + 8).astype(np.uint8)
    if len(unsigned) % 2:
        unsigned = np.append(unsigned, np.uint8(8))
    packed = unsigned[0::2] | (unsigned[1::2] << 4)
    return packed.tobytes()


def unpack_four_bit(payload: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    unsigned = np.empty(len(packed) * 2, dtype=np.uint8)
    unsigned[0::2] = packed & 0x0F
    unsigned[1::2] = packed >> 4
    return unsigned[:count].astype(np.int16).astype(np.int8) - 8


def serialize_state(state, bits: int) -> bytes:
    if bits not in (32, 8, 4):
        raise ValueError("bits must be 32, 8, or 4")
    buffer = io.BytesIO()
    buffer.write(struct.pack("<4sBH", SERIAL_MAGIC, bits, len(state)))
    for name, tensor in state.items():
        name_bytes = name.encode("utf-8")
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        buffer.write(struct.pack("<H", len(name_bytes)))
        buffer.write(name_bytes)
        buffer.write(struct.pack("<B", array.ndim))
        if array.ndim:
            buffer.write(struct.pack("<" + "I" * array.ndim, *array.shape))
        count = int(array.size)
        buffer.write(struct.pack("<I", count))
        if bits == 32:
            payload = array.tobytes(order="C")
            buffer.write(struct.pack("<I", len(payload)))
        else:
            qmax = 127 if bits == 8 else 7
            maximum = float(np.max(np.abs(array))) if count else 0.0
            scale = maximum / qmax if maximum > 0 else 1.0
            quantized = np.clip(np.rint(array.reshape(-1) / scale), -qmax, qmax).astype(np.int8)
            payload = quantized.tobytes() if bits == 8 else pack_four_bit(quantized)
            buffer.write(struct.pack("<fI", scale, len(payload)))
        buffer.write(payload)
    return buffer.getvalue()


def deserialize_state(payload: bytes):
    buffer = io.BytesIO(payload)
    magic, bits, number_of_tensors = struct.unpack("<4sBH", buffer.read(7))
    if magic != SERIAL_MAGIC:
        raise ValueError("Invalid model-message header.")
    state = {}
    for _ in range(number_of_tensors):
        (name_length,) = struct.unpack("<H", buffer.read(2))
        name = buffer.read(name_length).decode("utf-8")
        (number_of_dimensions,) = struct.unpack("<B", buffer.read(1))
        shape = (
            struct.unpack(
                "<" + "I" * number_of_dimensions,
                buffer.read(4 * number_of_dimensions),
            )
            if number_of_dimensions
            else ()
        )
        (count,) = struct.unpack("<I", buffer.read(4))
        if bits == 32:
            (byte_count,) = struct.unpack("<I", buffer.read(4))
            array = np.frombuffer(buffer.read(byte_count), dtype=np.float32).copy()
        else:
            scale, byte_count = struct.unpack("<fI", buffer.read(8))
            raw = buffer.read(byte_count)
            quantized = (
                np.frombuffer(raw, dtype=np.int8).copy()
                if bits == 8
                else unpack_four_bit(raw, count)
            )
            array = quantized.astype(np.float32) * scale
        state[name] = torch.from_numpy(array.reshape(shape).copy())
    if buffer.read(1):
        raise ValueError("Unexpected bytes after the model message.")
    return state


def weighted_average(states, sample_counts):
    weights = np.asarray(sample_counts, dtype=np.float64)
    weights /= weights.sum()
    result = {}
    for name in states[0]:
        result[name] = torch.stack(
            [state[name].float() * float(weight) for state, weight in zip(states, weights)]
        ).sum(0)
    return result

# %% [markdown]
# ## 6. Client-local training and evaluation

# %%
def clean_private_state(private_module):
    return {
        name.replace("_module.", "", 1): value.detach().cpu().clone()
        for name, value in private_module.state_dict().items()
    }


class FederatedClient:
    def __init__(self, client_id, images, labels):
        self.client_id = client_id
        (
            self.training_images,
            self.training_labels,
            self.validation_images,
            self.validation_labels,
        ) = split_local_train_validation(
            images,
            labels,
            VALIDATION_FRACTION,
            RUN_SEED + 10_000 + client_id,
        )
        self.training_features = None
        self.validation_features = None

    @property
    def training_size(self):
        return len(self.training_labels)

    def prepare_frozen_features(self, body_state, public_mean, public_std):
        # This method represents work performed on the client.
        if self.training_features is None:
            self.training_features = (
                extract_with_body(body_state, self.training_images) - public_mean
            ) / public_std
            self.validation_features = (
                extract_with_body(body_state, self.validation_images) - public_mean
            ) / public_std

    def train_local(
        self,
        method,
        global_payload,
        bits,
        learning_rate,
        noise_multiplier,
        max_grad_norm,
        accountant,
        local_seed,
        body_state,
        public_mean,
        public_std,
    ):
        seed_everything(local_seed)
        received_state = deserialize_state(global_payload)
        if method == "frozen":
            self.prepare_frozen_features(body_state, public_mean, public_std)
            model = nn.Linear(CNN.FEATURE_DIMENSION, PRIVATE_CLASSES)
            load_state(model, received_state)
            local_data = TensorDataset(self.training_features, self.training_labels)
        else:
            model = CNN(PRIVATE_CLASSES)
            load_state(model, received_state)
            local_data = TensorDataset(self.training_images, self.training_labels)

        loader = DataLoader(
            local_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=make_generator(local_seed + 1),
            num_workers=0,
        )
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        started = time.perf_counter()
        steps = 0
        loss_sum = 0.0

        if noise_multiplier is None:
            model.train()
            for _ in range(LOCAL_EPOCHS):
                for batch_inputs, batch_labels in loader:
                    optimizer.zero_grad()
                    loss = loss_function(
                        model(batch_inputs.to(DEVICE)),
                        batch_labels.to(DEVICE),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu())
                    steps += 1
            trained_state = cpu_state(model)
        else:
            privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
            private_model, private_optimizer, private_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                grad_sample_mode="hooks",
            )
            sample_rate = float(
                getattr(private_loader, "sample_rate", 1.0 / max(len(loader), 1))
            )
            private_model.train()
            for _ in range(LOCAL_EPOCHS):
                for batch_inputs, batch_labels in private_loader:
                    private_optimizer.zero_grad()
                    loss = loss_function(
                        private_model(batch_inputs.to(DEVICE)),
                        batch_labels.to(DEVICE),
                    )
                    loss.backward()
                    private_optimizer.step()
                    accountant.step(
                        noise_multiplier=noise_multiplier,
                        sample_rate=sample_rate,
                    )
                    loss_sum += float(loss.detach().cpu())
                    steps += 1
            trained_state = clean_private_state(private_model)

        upload_payload = serialize_state(trained_state, bits)
        return {
            "payload": upload_payload,
            "sample_count": self.training_size,
            "steps": steps,
            "mean_loss": loss_sum / max(steps, 1),
            "seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def validation_counts(
        self,
        method,
        state,
        body_state,
        public_mean,
        public_std,
    ):
        if method == "frozen":
            self.prepare_frozen_features(body_state, public_mean, public_std)
            model = nn.Linear(CNN.FEATURE_DIMENSION, PRIVATE_CLASSES).to(DEVICE)
            load_state(model, state)
            inputs = self.validation_features
        else:
            model = CNN(PRIVATE_CLASSES).to(DEVICE)
            load_state(model, state)
            inputs = self.validation_images
        model.eval()
        predictions = []
        for start in range(0, len(inputs), 512):
            predictions.append(model(inputs[start : start + 512].to(DEVICE)).argmax(1).cpu())
        predicted = torch.cat(predictions)
        correct = int((predicted == self.validation_labels).sum().item())
        return correct, len(self.validation_labels)


class EvaluationClient:
    def __init__(self, client_id, images, labels):
        self.client_id = client_id
        self.images = images
        self.labels = labels
        self.features = None

    def prepare_frozen_features(self, body_state, public_mean, public_std):
        if self.features is None:
            self.features = (
                extract_with_body(body_state, self.images) - public_mean
            ) / public_std

    @torch.no_grad()
    def confusion_counts(self, method, state, body_state, public_mean, public_std):
        if method == "frozen":
            self.prepare_frozen_features(body_state, public_mean, public_std)
            model = nn.Linear(CNN.FEATURE_DIMENSION, PRIVATE_CLASSES).to(DEVICE)
            load_state(model, state)
            inputs = self.features
        else:
            model = CNN(PRIVATE_CLASSES).to(DEVICE)
            load_state(model, state)
            inputs = self.images
        model.eval()
        confusion = np.zeros((PRIVATE_CLASSES, PRIVATE_CLASSES), dtype=np.int64)
        for start in range(0, len(inputs), 512):
            labels = self.labels[start : start + 512]
            predicted = model(inputs[start : start + 512].to(DEVICE)).argmax(1).cpu()
            np.add.at(confusion, (labels.numpy(), predicted.numpy()), 1)
        return confusion

# %% [markdown]
# ## 7. Prepare clients, fixed schedule, and initial models

# %%
print("Loading public and private datasets...")
public_images, public_labels = load_public(PUBLIC_CSV, PUBLIC_SAMPLE, RUN_SEED)
private_images, private_labels = load_emnist(EMNIST_TRAIN_CSV)
test_images, test_labels = load_emnist(EMNIST_TEST_CSV)

print(
    f"Public={len(public_labels):,} | Private train={len(private_labels):,} | "
    f"Official test={len(test_labels):,}"
)

if BODY_CHECKPOINT.exists():
    saved_body = torch.load(BODY_CHECKPOINT, map_location="cpu", weights_only=False)
    public_body_state = saved_body["body_state"]
    public_mean = saved_body["public_mean"]
    public_std = saved_body["public_std"]
    pretraining_seconds = float(saved_body["pretraining_seconds"])
    print("Loaded saved public body.")
else:
    public_body_state, public_mean, public_std, pretraining_seconds = pretrain_public_body(
        public_images, public_labels
    )
    torch.save(
        {
            "body_state": public_body_state,
            "public_mean": public_mean,
            "public_std": public_std,
            "pretraining_seconds": pretraining_seconds,
        },
        BODY_CHECKPOINT,
    )

del public_images, public_labels

training_partitions = dirichlet_partition(
    private_labels,
    NUM_CLIENTS,
    DIRICHLET_ALPHA,
    MIN_CLIENT_SIZE,
    RUN_SEED + 20_000,
)
test_partitions = balanced_test_partition(test_labels, NUM_CLIENTS, RUN_SEED + 30_000)

clients = [
    FederatedClient(
        client_id,
        private_images[indices].clone(),
        private_labels[indices].clone(),
    )
    for client_id, indices in enumerate(training_partitions)
]
evaluation_clients = [
    EvaluationClient(
        client_id,
        test_images[indices].clone(),
        test_labels[indices].clone(),
    )
    for client_id, indices in enumerate(test_partitions)
]
del private_images, private_labels, test_images, test_labels

schedule = build_schedule(
    ROUNDS,
    NUM_CLIENTS,
    CLIENTS_PER_ROUND,
    RUN_SEED + 40_000,
)

seed_everything(RUN_SEED + 50_000)
baseline_model = CNN(PRIVATE_CLASSES)
baseline_initial_state = cpu_state(baseline_model)

seed_everything(RUN_SEED + 60_000)
transfer_model = CNN(PRIVATE_CLASSES)
transfer_model.body.load_state_dict(public_body_state)
pretrained_initial_state = cpu_state(transfer_model)
frozen_initial_state = cpu_state(transfer_model.head)

INITIAL_STATES = {
    "baseline": baseline_initial_state,
    "pretrained": pretrained_initial_state,
    "frozen": frozen_initial_state,
}

partition_record = {
    "seed": RUN_SEED,
    "dirichlet_alpha": DIRICHLET_ALPHA,
    "clients": [indices.tolist() for indices in training_partitions],
    "local_training_sizes": [client.training_size for client in clients],
    "local_validation_sizes": [len(client.validation_labels) for client in clients],
    "official_test_sizes": [len(client.labels) for client in evaluation_clients],
}
save_json_atomic(OUTPUT_DIR / f"partition_seed_{RUN_SEED}.json", partition_record)
save_json_atomic(OUTPUT_DIR / f"schedule_seed_{RUN_SEED}.json", schedule)

print(
    "Local training sizes: "
    f"min={min(client.training_size for client in clients):,}, "
    f"mean={np.mean([client.training_size for client in clients]):.1f}, "
    f"max={max(client.training_size for client in clients):,}"
)

# %% [markdown]
# ## 8. Federated runner with per-round checkpoints

# %%
def classification_metrics(confusion):
    true_counts = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_counts,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_counts > 0,
    )
    recall = np.divide(
        true_positive,
        true_counts,
        out=np.zeros_like(true_positive, dtype=float),
        where=true_counts > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(confusion.sum())
    return {
        "accuracy_percent": 100.0 * float(true_positive.sum()) / max(total, 1),
        "macro_f1_percent": 100.0 * float(f1.mean()),
        "weighted_f1_percent": 100.0 * float(np.average(f1, weights=true_counts)),
        "total_examples": total,
        "confusion_matrix": confusion.tolist(),
    }


def privacy_summary(accountants, participation_counts):
    epsilons = []
    for client_id in range(NUM_CLIENTS):
        if participation_counts[client_id] == 0:
            epsilons.append(0.0)
        else:
            epsilons.append(float(accountants[client_id].get_epsilon(DELTA)))
    active = [value for value in epsilons if value > 0]
    return {
        "per_client_epsilon": epsilons,
        "max_epsilon": max(active) if active else 0.0,
        "mean_epsilon": float(np.mean(active)) if active else 0.0,
        "min_epsilon": min(active) if active else 0.0,
        "nonparticipating_clients": sum(value == 0 for value in epsilons),
    }


def checkpoint_name(run_key):
    safe = run_key.replace("|", "__").replace(".", "p")
    return CHECKPOINT_DIR / f"{safe}.pt"


def run_federated(
    run_key,
    method,
    bits,
    learning_rate,
    noise_multiplier,
    number_of_rounds,
    evaluate_test,
    evaluation_every,
):
    checkpoint_path = checkpoint_name(run_key)
    initial_state = copy.deepcopy(INITIAL_STATES[method])
    accountants = {client_id: RDPAccountant() for client_id in range(NUM_CLIENTS)}
    participation_counts = [0] * NUM_CLIENTS
    body_receivers = set()
    history = []
    training_upload_bytes = 0
    training_download_bytes = 0
    validation_metric_upload_bytes = 0
    start_round = 1
    global_state = initial_state
    run_seconds_before_resume = 0.0

    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        global_state = saved["global_state"]
        start_round = int(saved["completed_round"]) + 1
        participation_counts = list(saved["participation_counts"])
        body_receivers = set(saved["body_receivers"])
        history = list(saved["history"])
        training_upload_bytes = int(saved["training_upload_bytes"])
        training_download_bytes = int(saved["training_download_bytes"])
        validation_metric_upload_bytes = int(saved["validation_metric_upload_bytes"])
        run_seconds_before_resume = float(saved["run_seconds"])
        if noise_multiplier is not None:
            for client_id, state in saved["accountant_states"].items():
                accountants[int(client_id)].load_state_dict(state)
        print(f"Resuming {run_key} at round {start_round}.")

    body_payload_bytes = len(serialize_state(public_body_state, 32))
    started = time.perf_counter()

    for round_number in range(start_round, number_of_rounds + 1):
        selected_ids = schedule[round_number - 1]
        global_payload = serialize_state(global_state, bits)
        local_states = []
        local_sizes = []
        local_losses = []
        local_seconds = []
        round_upload = 0
        round_download = 0

        print(
            f"{run_key} | round {round_number}/{number_of_rounds} | "
            f"clients={selected_ids}"
        )

        for client_id in selected_ids:
            participation_counts[client_id] += 1
            round_download += len(global_payload)
            if method == "frozen" and client_id not in body_receivers:
                round_download += body_payload_bytes
                body_receivers.add(client_id)

            local_seed = (
                RUN_SEED * 1_000_000
                + round_number * 1_000
                + client_id
            )
            local_result = clients[client_id].train_local(
                method=method,
                global_payload=global_payload,
                bits=bits,
                learning_rate=learning_rate,
                noise_multiplier=noise_multiplier,
                max_grad_norm=MAX_GRAD_NORM,
                accountant=accountants[client_id],
                local_seed=local_seed,
                body_state=public_body_state,
                public_mean=public_mean,
                public_std=public_std,
            )
            round_upload += len(local_result["payload"])
            local_states.append(deserialize_state(local_result["payload"]))
            local_sizes.append(local_result["sample_count"])
            local_losses.append(local_result["mean_loss"])
            local_seconds.append(local_result["seconds"])

        global_state = weighted_average(local_states, local_sizes)
        training_upload_bytes += round_upload
        training_download_bytes += round_download

        should_evaluate = (
            round_number % evaluation_every == 0
            or round_number == number_of_rounds
        )
        if should_evaluate:
            # Evaluate the quantized state on selected clients' local validation data.
            evaluation_state = deserialize_state(serialize_state(global_state, bits))
            validation_correct = validation_total = 0
            for client_id in selected_ids:
                correct, total = clients[client_id].validation_counts(
                    method,
                    evaluation_state,
                    public_body_state,
                    public_mean,
                    public_std,
                )
                validation_correct += correct
                validation_total += total
                validation_metric_upload_bytes += 16  # two int64 counts

            privacy = (
                privacy_summary(accountants, participation_counts)
                if noise_multiplier is not None
                else None
            )
            history.append(
                {
                    "round": round_number,
                    "selected_clients": selected_ids,
                    "validation_accuracy_percent": (
                        100.0 * validation_correct / max(validation_total, 1)
                    ),
                    "mean_local_loss": float(np.mean(local_losses)),
                    "mean_local_seconds": float(np.mean(local_seconds)),
                    "round_upload_bytes": round_upload,
                    "round_download_bytes": round_download,
                    "privacy": privacy,
                }
            )
            print(
                f"  validation={history[-1]['validation_accuracy_percent']:.2f}% | "
                f"upload={round_upload / 1e6:.3f} MB | "
                f"download={round_download / 1e6:.3f} MB"
            )

        accountant_states = (
            {client_id: accountant.state_dict() for client_id, accountant in accountants.items()}
            if noise_multiplier is not None
            else {}
        )
        elapsed = run_seconds_before_resume + time.perf_counter() - started
        torch.save(
            {
                "global_state": global_state,
                "completed_round": round_number,
                "participation_counts": participation_counts,
                "body_receivers": sorted(body_receivers),
                "history": history,
                "training_upload_bytes": training_upload_bytes,
                "training_download_bytes": training_download_bytes,
                "validation_metric_upload_bytes": validation_metric_upload_bytes,
                "run_seconds": elapsed,
                "accountant_states": accountant_states,
            },
            checkpoint_path,
        )

    final_payload = serialize_state(global_state, bits)
    final_state = deserialize_state(final_payload)
    test_metrics = None
    evaluation_download_bytes = 0
    evaluation_metric_upload_bytes = 0

    if evaluate_test:
        aggregate_confusion = np.zeros(
            (PRIVATE_CLASSES, PRIVATE_CLASSES), dtype=np.int64
        )
        for evaluation_client in evaluation_clients:
            evaluation_download_bytes += len(final_payload)
            if method == "frozen":
                evaluation_download_bytes += body_payload_bytes
            local_confusion = evaluation_client.confusion_counts(
                method,
                final_state,
                public_body_state,
                public_mean,
                public_std,
            )
            evaluation_metric_upload_bytes += local_confusion.nbytes
            aggregate_confusion += local_confusion
        test_metrics = classification_metrics(aggregate_confusion)

    final_privacy = (
        privacy_summary(accountants, participation_counts)
        if noise_multiplier is not None
        else None
    )
    total_seconds = run_seconds_before_resume + time.perf_counter() - started
    result = {
        "run_key": run_key,
        "seed": RUN_SEED,
        "method": method,
        "bits": bits,
        "learning_rate": learning_rate,
        "noise_multiplier": noise_multiplier,
        "delta": DELTA if noise_multiplier is not None else None,
        "max_grad_norm": MAX_GRAD_NORM if noise_multiplier is not None else None,
        "rounds": number_of_rounds,
        "participation_counts": participation_counts,
        "history": history,
        "test": test_metrics,
        "privacy": final_privacy,
        "communication": {
            "training_upload_bytes": training_upload_bytes,
            "training_download_bytes": training_download_bytes,
            "training_total_bytes": training_upload_bytes + training_download_bytes,
            "validation_metric_upload_bytes": validation_metric_upload_bytes,
            "final_evaluation_download_bytes": evaluation_download_bytes,
            "final_evaluation_metric_upload_bytes": evaluation_metric_upload_bytes,
        },
        "runtime_seconds": total_seconds,
    }
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return result

# %% [markdown]
# ## 9. Validation-only learning-rate selection

# %%
if SELECTED_LR_JSON.exists():
    with SELECTED_LR_JSON.open("r", encoding="utf-8") as handle:
        selected_learning_rates = json.load(handle)
    print("Loaded selected learning rates:", selected_learning_rates)
else:
    selected_learning_rates = {}
    tuning_details = {}
    for method in METHODS:
        candidate_scores = {}
        for learning_rate in LR_CANDIDATES:
            tuning_key = f"tune|{method}|lr{learning_rate}"
            tuning_result = run_federated(
                run_key=tuning_key,
                method=method,
                bits=32,
                learning_rate=learning_rate,
                noise_multiplier=None,
                number_of_rounds=LR_TUNING_ROUNDS,
                evaluate_test=False,
                evaluation_every=1,
            )
            recent_scores = [
                item["validation_accuracy_percent"]
                for item in tuning_result["history"][-2:]
            ]
            candidate_scores[str(learning_rate)] = float(np.mean(recent_scores))
        best_learning_rate = max(
            LR_CANDIDATES,
            key=lambda value: (candidate_scores[str(value)], -value),
        )
        selected_learning_rates[method] = best_learning_rate
        tuning_details[method] = candidate_scores
        print(
            f"{method}: selected LR={best_learning_rate} "
            f"from validation scores {candidate_scores}"
        )
    save_json_atomic(
        SELECTED_LR_JSON,
        {
            "seed": RUN_SEED,
            "selected_learning_rates": selected_learning_rates,
            "validation_scores": tuning_details,
            "test_data_used": False,
        },
    )

if "selected_learning_rates" in selected_learning_rates:
    selected_learning_rates = selected_learning_rates["selected_learning_rates"]

# %% [markdown]
# ## 10. Complete experiment matrix

# %%
base_results = {
    "metadata": {
        "seed": RUN_SEED,
        "device": str(DEVICE),
        "public_dataset": str(PUBLIC_CSV),
        "private_train_dataset": str(EMNIST_TRAIN_CSV),
        "private_test_dataset": str(EMNIST_TEST_CSV),
        "number_of_clients": NUM_CLIENTS,
        "clients_per_round": CLIENTS_PER_ROUND,
        "participation_fraction": CLIENTS_PER_ROUND / NUM_CLIENTS,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "validation_fraction": VALIDATION_FRACTION,
        "privacy_settings": PRIVACY_SETTINGS,
        "bits_by_method": BITS_BY_METHOD,
        "selected_learning_rates": selected_learning_rates,
        "public_pretraining_seconds": pretraining_seconds,
        "test_policy": "Official test used only after each preregistered configuration finishes.",
        "privacy_scope": "Example-level DP within participating clients.",
    },
    "experiments": {},
}

if RESULTS_JSON.exists():
    with RESULTS_JSON.open("r", encoding="utf-8") as handle:
        saved_results = json.load(handle)
    base_results["experiments"].update(saved_results.get("experiments", {}))
    print(f"Loaded {len(base_results['experiments'])} completed configurations.")

for privacy_label, noise_multiplier in PRIVACY_SETTINGS.items():
    for method in METHODS:
        for bits in BITS_BY_METHOD[method]:
            run_key = f"{method}|{privacy_label}|{bits}bit"
            if run_key in base_results["experiments"]:
                print(f"Skipping completed configuration: {run_key}")
                continue
            print(f"\n=== {run_key} ===")
            result = run_federated(
                run_key=run_key,
                method=method,
                bits=bits,
                learning_rate=float(selected_learning_rates[method]),
                noise_multiplier=noise_multiplier,
                number_of_rounds=ROUNDS,
                evaluate_test=True,
                evaluation_every=EVALUATE_EVERY,
            )
            base_results["experiments"][run_key] = result
            save_json_atomic(RESULTS_JSON, base_results)
            print(
                f"Completed {run_key}: "
                f"test accuracy={result['test']['accuracy_percent']:.2f}%"
            )

# %% [markdown]
# ## 11. Tables, figures, and downloadable ZIP

# %%
summary_rows = []
privacy_rows = []
for run_key, result in base_results["experiments"].items():
    communication = result["communication"]
    privacy = result["privacy"]
    summary_rows.append(
        {
            "run_key": run_key,
            "seed": RUN_SEED,
            "method": result["method"],
            "noise_multiplier": result["noise_multiplier"],
            "bits": result["bits"],
            "learning_rate": result["learning_rate"],
            "test_accuracy_percent": result["test"]["accuracy_percent"],
            "macro_f1_percent": result["test"]["macro_f1_percent"],
            "weighted_f1_percent": result["test"]["weighted_f1_percent"],
            "epsilon_max": privacy["max_epsilon"] if privacy else np.nan,
            "epsilon_mean": privacy["mean_epsilon"] if privacy else np.nan,
            "epsilon_min": privacy["min_epsilon"] if privacy else np.nan,
            "training_upload_mb": communication["training_upload_bytes"] / 1e6,
            "training_download_mb": communication["training_download_bytes"] / 1e6,
            "training_total_mb": communication["training_total_bytes"] / 1e6,
            "runtime_minutes": result["runtime_seconds"] / 60.0,
        }
    )
    if privacy:
        for client_id, epsilon in enumerate(privacy["per_client_epsilon"]):
            privacy_rows.append(
                {
                    "run_key": run_key,
                    "seed": RUN_SEED,
                    "client_id": client_id,
                    "participations": result["participation_counts"][client_id],
                    "epsilon": epsilon,
                }
            )

summary_frame = pd.DataFrame(summary_rows).sort_values(
    ["noise_multiplier", "method", "bits"], na_position="first"
)
privacy_frame = pd.DataFrame(privacy_rows)
summary_csv = OUTPUT_DIR / f"summary_seed_{RUN_SEED}.csv"
privacy_csv = OUTPUT_DIR / f"client_privacy_seed_{RUN_SEED}.csv"
summary_frame.to_csv(summary_csv, index=False)
privacy_frame.to_csv(privacy_csv, index=False)
display(summary_frame)

# %%
plt.style.use("seaborn-v0_8-whitegrid")
colors = {"baseline": "#3366CC", "pretrained": "#109618", "frozen": "#DC3912"}

# Privacy-accuracy trade-off at 32-bit.
fig, axis = plt.subplots(figsize=(7.2, 4.4))
for method in METHODS:
    subset = summary_frame[(summary_frame.method == method) & (summary_frame.bits == 32)].copy()
    subset["sigma_plot"] = subset["noise_multiplier"].fillna(0.0)
    subset = subset.sort_values("sigma_plot")
    axis.plot(
        subset["sigma_plot"],
        subset["test_accuracy_percent"],
        marker="o",
        label=method.title(),
        color=colors[method],
    )
axis.set_xlabel("Noise multiplier (0 = No-DP)")
axis.set_ylabel("Test accuracy (%)")
axis.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "privacy_accuracy_tradeoff.png", dpi=300)
plt.show()

# %%
# Convergence for sigma=1.0 at 32-bit.
fig, axis = plt.subplots(figsize=(7.2, 4.4))
for method in METHODS:
    key = f"{method}|sigma1.0|32bit"
    history = base_results["experiments"][key]["history"]
    axis.plot(
        [item["round"] for item in history],
        [item["validation_accuracy_percent"] for item in history],
        marker="o",
        label=method.title(),
        color=colors[method],
    )
axis.set_xlabel("Federated round")
axis.set_ylabel("Local validation accuracy (%)")
axis.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "convergence_sigma1.png", dpi=300)
plt.show()

# %%
# Quantization effect for the proposed frozen method.
fig, axis = plt.subplots(figsize=(7.2, 4.4))
for privacy_label, noise_multiplier in PRIVACY_SETTINGS.items():
    subset = summary_frame[
        (summary_frame.method == "frozen")
        & (
            summary_frame.noise_multiplier.isna()
            if noise_multiplier is None
            else summary_frame.noise_multiplier.eq(noise_multiplier)
        )
    ].sort_values("bits")
    axis.plot(
        subset["bits"],
        subset["test_accuracy_percent"],
        marker="o",
        label=privacy_label,
    )
axis.set_xticks([4, 8, 32])
axis.set_xlabel("Message precision (bits)")
axis.set_ylabel("Test accuracy (%)")
axis.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "quantization_accuracy.png", dpi=300)
plt.show()

# %%
# Measured serialized training communication at sigma=1.0.
communication_subset = summary_frame[summary_frame.noise_multiplier.eq(1.0)].copy()
communication_subset["label"] = (
    communication_subset["method"].str.title()
    + " "
    + communication_subset["bits"].astype(str)
    + "-bit"
)
fig, axis = plt.subplots(figsize=(8.0, 4.6))
axis.bar(
    communication_subset["label"],
    communication_subset["training_total_mb"],
    color=[colors[item] for item in communication_subset["method"]],
)
axis.set_ylabel("Serialized training communication (MB)")
axis.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "communication_sigma1.png", dpi=300)
plt.show()

# %%
# Client participation and epsilon for the main proposed DP setting.
main_key = "frozen|sigma1.0|32bit"
main_result = base_results["experiments"][main_key]
fig, left_axis = plt.subplots(figsize=(8.0, 4.6))
client_ids = np.arange(NUM_CLIENTS)
left_axis.bar(client_ids, main_result["participation_counts"], color="#3366CC", alpha=0.7)
left_axis.set_xlabel("Client ID")
left_axis.set_ylabel("Rounds participated", color="#3366CC")
right_axis = left_axis.twinx()
right_axis.plot(
    client_ids,
    main_result["privacy"]["per_client_epsilon"],
    color="#DC3912",
    marker=".",
    linewidth=1,
)
right_axis.set_ylabel("Final epsilon", color="#DC3912")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "client_participation_privacy.png", dpi=300)
plt.show()

# %%
config_record = dict(base_results["metadata"])
config_record["output_files"] = {
    "results": RESULTS_JSON.name,
    "summary": summary_csv.name,
    "client_privacy": privacy_csv.name,
}
save_json_atomic(OUTPUT_DIR / f"config_seed_{RUN_SEED}.json", config_record)

zip_base = Path(f"/kaggle/working/pipeline1_final_seed_{RUN_SEED}")
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_DIR)

print("\nFINAL SEED RUN COMPLETE")
print(f"Results folder: {OUTPUT_DIR}")
print(f"Download ZIP: {zip_path}")
