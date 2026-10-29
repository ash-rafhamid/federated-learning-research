# Federated Learning Research

Repository for federated-learning research code. The current implementation is Pipeline 1, described below. Additional experiment pipelines can be added to this repository as they become ready.

Research code for **Frozen Public Representations for Communication-Efficient Differentially Private Federated Learning**. The experiment compares training an entire convolutional network with training only a classifier on frozen public features, using non-IID EMNIST clients, DP-SGD, and measured serialized model messages.

The repository contains the corrected experiment scripts, three ready-to-run Kaggle notebooks, a notebook for combining their results, and recorded result tables. The training code is preserved from the completed research runs; preparing this repository does not rerun training.

## What the experiment compares

| Method | Initialization and private training | Trainable parameters | Message precision |
| --- | --- | ---: | --- |
| Baseline | Random CNN; train the full model | 99,183 | 32-bit |
| Pretrained | Publicly pretrained body; fine-tune the full model | 99,183 | 32-bit |
| Frozen | Fix the public body; train the classifier only | 6,063 | 32-, 8-, and packed 4-bit |

Each seed runs **20 main configurations**: five method/precision combinations across No-DP and noise multipliers `0.5`, `1.0`, and `2.0`. Seeds `42`, `123`, and `2026` give **60 main runs**, in addition to public pretraining and learning-rate tuning.

## Run on Kaggle

1. Import one of these notebooks into Kaggle:

   | Session | Notebook |
   | --- | --- |
   | Seed 42 | [Pipeline_1_Final_Seed_42_Kaggle.ipynb](Pipeline_1_Final_Seed_42_Kaggle.ipynb) |
   | Seed 123 | [Pipeline_1_Final_Seed_123_Kaggle.ipynb](Pipeline_1_Final_Seed_123_Kaggle.ipynb) |
   | Seed 2026 | [Pipeline_1_Final_Seed_2026_Kaggle.ipynb](Pipeline_1_Final_Seed_2026_Kaggle.ipynb) |

2. Attach both datasets below. A separate MNIST dataset is **not used** by this corrected pipeline.
3. Enable a GPU accelerator. Enable Internet for the first cell if Opacus must be installed.
4. Check the three CSV paths in the configuration cell against Kaggle's attached input paths. Mount paths can differ between notebooks.
5. Run every cell from top to bottom. Run the three seeds in separate sessions if needed; keep every other experimental setting the same.
6. Download `/kaggle/working/pipeline1_final_seed_<SEED>.zip` after each session finishes.

The script uses a single `cuda` device when available and falls back to CPU. It does not distribute training over two GPUs automatically. CSV loading also needs host RAM. Runtime depends on the accelerator, environment, and cached work; there is no fixed completion-time guarantee.

### Required datasets

Both sources contain **28 x 28 grayscale images of handwritten characters**, stored as rows of pixel values in CSV files. Each image has 784 pixel values and a label.

| Role | Source | File(s) used | Task |
| --- | --- | --- | --- |
| Public pretraining | [Digit and English Character Dataset](https://www.kaggle.com/datasets/swarupshah/digit-character) | `digit_char_splited_dataset.csv` | Sample 60,000 images; 36 classes: digits 0-9 and uppercase A-Z |
| Private FL benchmark | [EMNIST](https://www.kaggle.com/datasets/crawford/emnist) | `emnist-balanced-train.csv`, `emnist-balanced-test.csv` | 47 classes; 112,800 official training images and 18,800 official test images |

"Public" and "private" describe their experimental roles. EMNIST itself is a publicly available benchmark; no real patient records are used. Dataset files are obtained from their providers and are not included in this repository.

The default paths are:

```text
/kaggle/input/datasets/swarupshah/digit-character/digit_char_splited_dataset.csv
/kaggle/input/datasets/crawford/emnist/emnist-balanced-train.csv
/kaggle/input/datasets/crawford/emnist/emnist-balanced-test.csv
```

## Experimental protocol

| Setting | Value |
| --- | --- |
| Federation | 50 simulated clients; 15 sampled uniformly without replacement per round |
| Training schedule | 20 rounds; 3 local epochs; batch size 32 |
| Non-IID partition | Dirichlet alpha = 0.5; minimum 128 examples per client before splitting |
| Local split | Approximately 90% training / 10% validation per client |
| Public model | Three Conv-GroupNorm-ReLU blocks; 128-dimensional features; five pretraining epochs |
| Private optimizer | SGD with momentum 0.9 |
| Learning-rate selection | Candidates 0.3 and 0.5; five No-DP rounds; local validation only |
| Privacy | No-DP control; sigma = 0.5, 1.0, 2.0; clipping norm C = 1.0; delta = 0.00001 |
| Checkpoints for reporting | Rounds 5, 10, 15, and 20 |
| Replication | Seeds 42, 123, and 2026; mean and sample standard deviation |

For each seed, the pipeline samples public data, pretrains the public body, creates client partitions and train/validation splits, selects learning rates, and runs the full comparison. Configurations within a seed reuse the same client-selection schedule. Different seeds regenerate the pipeline, including public pretraining and partitioning.

Frozen training and validation features are computed and cached inside the simulated client objects using normalization statistics estimated from public features. FedAvg weights each participating client's model by its training-set size. The official test set is evaluated after each fixed main configuration finishes, using aggregated client confusion matrices; it is not used for learning-rate selection.

See [the detailed protocol](docs/EXPERIMENT_PROTOCOL.md) for accounting, serialization, resume behavior, and interpretation of measurements.

## Outputs from each seed

The run writes `/kaggle/working/pipeline1_final_seed_<SEED>/` and a ZIP with the same base name.

| Output | Contents |
| --- | --- |
| `results_seed_<SEED>.json` | Main configuration results, reporting history, participation counts, privacy, and byte counts |
| `summary_seed_<SEED>.csv` | Accuracy, macro-F1, weighted-F1, epsilon summaries, communication, and runtime |
| `client_privacy_seed_<SEED>.csv` | Final epsilon and participation count for each client in each DP configuration |
| `selected_learning_rates_seed_<SEED>.json` | Selected rates and validation tuning scores |
| `partition_seed_<SEED>.json`, `schedule_seed_<SEED>.json` | Partition/split records and the client-selection schedule |
| `config_seed_<SEED>.json` | Experiment configuration and output filenames |
| `public_body_seed_<SEED>.pt` | Public body and public feature normalization statistics |
| `figures/` | Accuracy, convergence, quantization, communication, and participation/privacy plots |
| `checkpoints/` | In-progress round checkpoints; removed for a configuration when it completes |

Round checkpoints support resuming interrupted work **when the existing output directory is preserved**. Kaggle session storage is not automatically restored by this code. See the protocol before reusing previous output files.

## Combine the three seeds

1. Upload the three `pipeline1_final_seed_*.zip` files as inputs to a new Kaggle notebook. Keep these filenames so automatic discovery can find them.
2. Import [Pipeline_1_Final_Combine_Three_Seeds_Kaggle.ipynb](Pipeline_1_Final_Combine_Three_Seeds_Kaggle.ipynb).
3. Run all cells. The combiner expects seeds 42, 123, and 2026 and checks that their configuration-key sets match. Supply completed runs with 20 configurations each.
4. Download `/kaggle/working/pipeline1_three_seed_summary.zip`.

The combined output includes `all_seed_results.csv`, `three_seed_mean_std.csv`, `all_client_privacy.csv`, a manifest, and figures. Standard deviation is the sample standard deviation across seeds.

## Recorded results

The files in [results/](results/) are the saved tables from the completed three-seed research runs. They are included for inspection and were not regenerated when this repository was prepared. Their report-oriented column names differ from the combiner's native CSV schema; [the schema guide](docs/RESULTS.md) explains the mapping.

At **sigma = 1.0**:

| Method | Bits | Test accuracy, mean +/- sample SD (%) | Training communication (MB) |
| --- | ---: | ---: | ---: |
| Baseline | 32 | 11.97 +/- 4.11 | 238.292 |
| Pretrained | 32 | 24.08 +/- 8.58 | 238.292 |
| Frozen | 32 | 77.63 +/- 0.63 | 33.221 |
| Frozen | 8 | 77.64 +/- 0.64 | 22.313 |
| Frozen | 4 | 74.46 +/- 1.07 | 20.494 |

Frozen 8-bit reduces recorded training traffic by approximately **90.64%** relative to full-model exchange. These results describe this benchmark and protocol. Public initialization performs best in the No-DP comparison; freezing is not uniformly best in every setting.

## Local inspection and dependencies

Kaggle is the primary execution environment. For local inspection:

```bash
git clone https://github.com/ash-rafhamid/federated-learning-research.git
cd federated-learning-research
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_repository.py
```

The repository checker uses only the Python standard library and does not train models or load datasets. The `.py` files use `# %%` cell markers and match the notebook code; the generic experiment script defaults to `RUN_SEED = 42`.

For local execution, change the dataset and output paths in the configuration cell. For the combiner, also change `INPUT_ROOTS`, `OUTPUT_DIR`, and the archive output path. The scripts execute the pipeline at top level, so importing them as libraries starts work.

`requirements.txt` lists direct dependencies. Historical Kaggle package versions were not saved as a complete environment lockfile, so this is not a claim of bit-for-bit reproducibility across all library or hardware versions. Use a CUDA-compatible PyTorch installation when running on a local GPU.

## Privacy and measurement scope

- Clients and server are conceptual objects in one Python process. This is an FL simulation, not a deployment on isolated physical devices. The program initially loads the benchmark centrally before constructing clients.
- Reported epsilon accounts for local DP-SGD training steps within participating clients at the specified delta. It is example-level accounting, not a client-participation guarantee; secure aggregation is not implemented. Opacus runs with `secure_mode=False` for these reproducible experiments.
- Validation scores, hyperparameter selection, confusion matrices, and metadata have no additional privacy mechanism here. The reported training epsilon must not be described as covering every possible data-dependent output. If several models trained on the same private records are released, their privacy costs require composition beyond a single configuration's ledger.
- Communication means actual serialized **model-message bytes** for training uploads and downloads, including serialization metadata and a one-time public-body download for each participating Frozen client. It excludes network framing, cryptography, and validation/final-evaluation traffic, which is recorded separately. MB means 1,000,000 bytes.
- Measured configuration runtime reflects this process, its caches, and evaluation work. Public pretraining is recorded separately. Three seeds describe variation in this experiment; they do not establish performance across arbitrary tasks or deployments.

## Repository contents

```text
.
|-- Pipeline_1_Final_Corrected_One_Seed_Kaggle.py
|-- Pipeline_1_Final_Seed_42_Kaggle.ipynb
|-- Pipeline_1_Final_Seed_123_Kaggle.ipynb
|-- Pipeline_1_Final_Seed_2026_Kaggle.ipynb
|-- Pipeline_1_Final_Combine_Three_Seeds_Kaggle.py
|-- Pipeline_1_Final_Combine_Three_Seeds_Kaggle.ipynb
|-- docs/
|   |-- EXPERIMENT_PROTOCOL.md
|   `-- RESULTS.md
|-- results/
|   |-- three_seed_all_results.csv
|   |-- three_seed_summary.csv
|   `-- derived_statistics.csv
|-- tools/check_repository.py
`-- requirements.txt
```
