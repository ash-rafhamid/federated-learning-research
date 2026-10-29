# %% [markdown]
# # Pipeline 1: combine the three completed seed runs
# Upload the three output ZIP files from seeds 42, 123, and 2026, then run all cells.

# %%
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# %% [markdown]
# ## 1. Locate and extract the three results

# %%
EXPECTED_SEEDS = {42, 123, 2026}
INPUT_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working")]
OUTPUT_DIR = Path("/kaggle/working/pipeline1_three_seed_summary")
EXTRACT_DIR = OUTPUT_DIR / "extracted_inputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

zip_paths = []
for root in INPUT_ROOTS:
    if root.exists():
        zip_paths.extend(root.rglob("pipeline1_final_seed_*.zip"))

for zip_path in sorted(set(zip_paths)):
    destination = EXTRACT_DIR / zip_path.stem
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    print(f"Extracted: {zip_path}")

result_paths = []
for root in [*INPUT_ROOTS, EXTRACT_DIR]:
    if root.exists():
        result_paths.extend(root.rglob("results_seed_*.json"))

results_by_seed = {}
for result_path in sorted(set(result_paths)):
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    seed = int(result["metadata"]["seed"])
    if seed in EXPECTED_SEEDS:
        results_by_seed[seed] = result
        print(f"Found seed {seed}: {result_path}")

missing = EXPECTED_SEEDS - set(results_by_seed)
if missing:
    raise FileNotFoundError(
        f"Missing completed result files for seeds: {sorted(missing)}"
    )

# %% [markdown]
# ## 2. Verify that the experiment matrices match

# %%
reference_keys = set(results_by_seed[42]["experiments"])
for seed, result in results_by_seed.items():
    current_keys = set(result["experiments"])
    if current_keys != reference_keys:
        missing_keys = reference_keys - current_keys
        extra_keys = current_keys - reference_keys
        raise ValueError(
            f"Seed {seed} has a different experiment matrix. "
            f"Missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
        )

print(f"Verified {len(reference_keys)} configurations for every seed.")

# %% [markdown]
# ## 3. Build per-seed and mean ± standard-deviation tables

# %%
rows = []
privacy_rows = []
for seed, result_file in sorted(results_by_seed.items()):
    for run_key, result in result_file["experiments"].items():
        noise = result["noise_multiplier"]
        privacy_label = "NoDP" if noise is None else f"sigma{noise}"
        communication = result["communication"]
        privacy = result["privacy"]
        rows.append(
            {
                "seed": seed,
                "run_key": run_key,
                "method": result["method"],
                "privacy_label": privacy_label,
                "noise_multiplier": noise,
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
                        "seed": seed,
                        "run_key": run_key,
                        "client_id": client_id,
                        "participations": result["participation_counts"][client_id],
                        "epsilon": epsilon,
                    }
                )

per_seed = pd.DataFrame(rows)
client_privacy = pd.DataFrame(privacy_rows)
group_columns = ["method", "privacy_label", "noise_multiplier", "bits"]
metric_columns = [
    "test_accuracy_percent",
    "macro_f1_percent",
    "weighted_f1_percent",
    "epsilon_max",
    "epsilon_mean",
    "epsilon_min",
    "training_upload_mb",
    "training_download_mb",
    "training_total_mb",
    "runtime_minutes",
]

combined = (
    per_seed.groupby(group_columns, dropna=False)[metric_columns]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
combined.columns = [
    "_".join(str(item) for item in column if str(item))
    if isinstance(column, tuple)
    else str(column)
    for column in combined.columns
]

per_seed.to_csv(OUTPUT_DIR / "all_seed_results.csv", index=False)
combined.to_csv(OUTPUT_DIR / "three_seed_mean_std.csv", index=False)
client_privacy.to_csv(OUTPUT_DIR / "all_client_privacy.csv", index=False)

display(per_seed.sort_values(["privacy_label", "method", "bits"]))
display(combined)

# %% [markdown]
# ## 4. Publication-ready three-seed figures

# %%
plt.style.use("seaborn-v0_8-whitegrid")
colors = {"baseline": "#3366CC", "pretrained": "#109618", "frozen": "#DC3912"}
sigma_order = [0.0, 0.5, 1.0, 2.0]

fig, axis = plt.subplots(figsize=(7.4, 4.6))
for method in ["baseline", "pretrained", "frozen"]:
    subset = per_seed[(per_seed.method == method) & (per_seed.bits == 32)].copy()
    subset["sigma_plot"] = subset["noise_multiplier"].fillna(0.0)
    grouped = subset.groupby("sigma_plot")["test_accuracy_percent"].agg(["mean", "std"])
    grouped = grouped.reindex(sigma_order)
    axis.errorbar(
        grouped.index,
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=4,
        label=method.title(),
        color=colors[method],
    )
axis.set_xlabel("Noise multiplier (0 = No-DP)")
axis.set_ylabel("Test accuracy, mean ± SD (%)")
axis.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "three_seed_privacy_accuracy.png", dpi=300)
plt.show()

# %%
fig, axis = plt.subplots(figsize=(7.4, 4.6))
subset = per_seed[
    (per_seed.method == "frozen")
    & (per_seed.privacy_label == "sigma1.0")
]
grouped = subset.groupby("bits")["test_accuracy_percent"].agg(["mean", "std"])
grouped = grouped.reindex([4, 8, 32])
axis.errorbar(
    grouped.index,
    grouped["mean"],
    yerr=grouped["std"],
    marker="o",
    capsize=4,
    color=colors["frozen"],
)
axis.set_xticks([4, 8, 32])
axis.set_xlabel("Message precision (bits)")
axis.set_ylabel("Test accuracy, mean ± SD (%)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "three_seed_quantization_accuracy.png", dpi=300)
plt.show()

# %%
communication_subset = per_seed[per_seed.privacy_label == "sigma1.0"].copy()
communication_summary = (
    communication_subset.groupby(["method", "bits"])["training_total_mb"]
    .agg(["mean", "std"])
    .reset_index()
)
communication_summary["label"] = (
    communication_summary["method"].str.title()
    + " "
    + communication_summary["bits"].astype(str)
    + "-bit"
)
fig, axis = plt.subplots(figsize=(8.2, 4.8))
axis.bar(
    communication_summary["label"],
    communication_summary["mean"],
    yerr=communication_summary["std"],
    capsize=4,
    color=[colors[item] for item in communication_summary["method"]],
)
axis.set_ylabel("Serialized training communication, mean ± SD (MB)")
axis.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "three_seed_communication.png", dpi=300)
plt.show()

# %%
runtime_by_seed = per_seed.groupby("seed")["runtime_minutes"].sum().reset_index()
fig, axis = plt.subplots(figsize=(6.8, 4.4))
axis.bar(runtime_by_seed["seed"].astype(str), runtime_by_seed["runtime_minutes"], color="#666666")
axis.set_xlabel("Seed")
axis.set_ylabel("Total experiment runtime (minutes)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "runtime_by_seed.png", dpi=300)
plt.show()

# %% [markdown]
# ## 5. Save the combined package

# %%
manifest = {
    "seeds": sorted(EXPECTED_SEEDS),
    "configurations_per_seed": len(reference_keys),
    "accuracy_report": "Mean and sample standard deviation across three seeds.",
    "source_result_files": [
        f"results_seed_{seed}.json" for seed in sorted(EXPECTED_SEEDS)
    ],
}
with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

zip_path = shutil.make_archive(
    "/kaggle/working/pipeline1_three_seed_summary",
    "zip",
    root_dir=OUTPUT_DIR,
)
print(f"Combined folder: {OUTPUT_DIR}")
print(f"Download ZIP: {zip_path}")
