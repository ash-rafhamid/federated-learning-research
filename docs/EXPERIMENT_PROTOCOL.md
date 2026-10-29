# Experiment protocol

This document describes the code shipped in this repository. The scripts retain the completed experiment implementation; the repository preparation did not retrain or change the models.

## 1. Public representation

Each seed samples 60,000 images without replacement from `digit_char_splited_dataset.csv`. Public labels run from 0 to 35. The model has three convolutional blocks with 32, 64, and 128 channels, Group Normalization and ReLU, two max-pooling layers, and adaptive average pooling. A temporary 36-class head is used during five public pretraining epochs.

The temporary head is discarded. The convolutional body has 93,120 parameters. Public features provide the mean and standard deviation used to normalize Frozen features; private records do not determine those statistics.

## 2. Private clients and evaluation split

The EMNIST Balanced training CSV supplies 112,800 records and 47 labels, from 0 to 46. Its 18,800-record official test split stays separate. Images are transposed to correct CSV orientation and scaled to [0, 1].

Training records are partitioned among 50 clients by class-wise Dirichlet draws with alpha 0.5. Partitions are retried if any client has fewer than 128 examples. Each accepted client makes a deterministic approximately 90/10 training/validation split. The minimum applies before splitting. The official test set is assigned once across evaluation clients.

The split records and participation schedule are saved. In each round, 15 clients are sampled uniformly without replacement. Within a seed, every configuration uses the same schedule. This is fixed-size client sampling; Opacus separately uses its private minibatch-sampling mechanism within selected clients.

## 3. Three training regimes

Baseline starts with a randomly initialized full 47-class network. Pretrained initializes the body with public weights and trains the full network. Both have 99,183 trainable parameters and exchange 32-bit states.

Frozen keeps the public body fixed and trains a 128-input, 47-output linear classifier with 6,063 parameters. Each client caches its training and validation features separately. Only the classifier state is repeatedly exchanged; the byte ledger charges a public-body download once per participating client for each configuration.

Full-model training uses client images directly. Frozen feature extraction occurs through client methods in the simulation. These object boundaries represent the intended FL organization, but all objects run within one process and do not create operating-system or cryptographic isolation.

## 4. Learning rates and main runs

The candidates are 0.3 and 0.5. Each regime runs five No-DP tuning rounds per candidate, evaluates validation accuracy every tuning round, and averages the final two validation accuracies to score the candidate. The selected learning rate is saved and reused across that regime's main privacy/precision settings. The official test set is not used in this selection.

Each seed runs the following matrix for 20 rounds:

| Method | Bits | Privacy settings |
| --- | --- | --- |
| Baseline | 32 | NoDP, sigma0.5, sigma1.0, sigma2.0 |
| Pretrained | 32 | NoDP, sigma0.5, sigma1.0, sigma2.0 |
| Frozen | 32, 8, 4 | NoDP, sigma0.5, sigma1.0, sigma2.0 |

There are 20 main configurations per seed and 60 across the three seeds. Each configuration has 300 client selections. Local optimization uses three epochs, batch size 32, and SGD with momentum 0.9. Full-model states or classifier states are weighted by client training-set size during FedAvg.

## 5. Privacy ledger

DP configurations use per-example clipping at C = 1.0 and Gaussian noise multipliers 0.5, 1.0, or 2.0. Each client's external RDP accountant persists throughout that configuration and advances with the actual private-loader sampling rate at each local optimizer step. It is restored from round checkpoints and does not reset at round boundaries.

At delta = 0.00001, the code reports maximum epsilon, the mean over participating clients, and the minimum over participating clients. Clients that never participate have zero accounted training steps and are counted separately. The maximum is the conservative single-run summary across clients. Validation/privacy reporting occurs at rounds 5, 10, 15, and 20. Per-client final epsilon values and participation counts are saved.

This ledger concerns example-level DP-SGD training within clients. It does not provide client-level participation privacy, secure aggregation, or a privacy guarantee for unprotected validation statistics and model-selection decisions. Multiple released configurations trained on the same records need further composition. `secure_mode=False` is used in the archived experiment.

## 6. Serialization and communication

The implementation really serializes and deserializes model states. Full models use 32-bit floating-point payloads. Frozen heads additionally support symmetric signed 8-bit integers and packed signed 4-bit integers. Messages carry tensor names, shapes, scales, and payloads. Quantization is performed after the optimizer update.

Training communication is the sum of uploaded and downloaded model bytes. A selected Frozen client is also charged for receiving the public body once per configuration. Validation count uploads and final-evaluation model/confusion-matrix traffic are tracked in separate fields, outside the training total. The protocol does not model network frames, packet loss, network latency, cryptography, or transfer of every auxiliary item a physical deployment would need. All MB values use decimal bytes / 1,000,000.

## 7. Evaluation, statistics, and runtime

Each fixed main configuration is evaluated on the official test split after its 20 rounds finish. Evaluation clients compute confusion matrices, which are summed to obtain accuracy, macro-F1, and weighted-F1. Test results are not used by the code to tune the learning rate or stop training early.

Seeds 42, 123, and 2026 regenerate public data sampling, pretraining, client partitions/splits, initialization, and schedules. Pandas aggregation reports the arithmetic mean and sample standard deviation across seeds. The combiner checks matching key sets, but matching incomplete inputs can also pass that check; confirm that each input contains all 20 expected main configurations.

`runtime_seconds` measures a configuration, including its reporting and final evaluation, rather than total session wall time. Public pretraining time is stored separately; tuning, loading, shared cached feature work, plotting, and package creation also affect session duration. Cached features are reused, so per-configuration times should be interpreted in execution order. They do not measure latency or speedup on a physical federated network.

## 8. Checkpoints and resuming

The runner saves its global state, completed round, participation counts, histories, message counters, elapsed time, and accountant states after each round. It saves completed configurations to JSON and skips those keys on a rerun. The public body and selected learning rates are also reused when saved files exist. An in-progress checkpoint is deleted after that configuration completes.

Resuming requires the previous files to remain at the configured paths. A fresh Kaggle session needs the previous output directory restored before execution. After changing a seed, partition, hyperparameter, dataset, or implementation, use a fresh output directory: saved artifacts are selected by filename/run key and are not fully validated against every changed setting.

## 9. Environment

The original code installs Opacus only when it is absent and uses whatever compatible library versions are available in the Kaggle environment. A historical environment lockfile is not available. `requirements.txt` lists direct dependencies without claiming those are the historical versions. The GPU path uses one CUDA device; clients execute sequentially.
