# Recorded result tables

The three CSV files in `results/` were copied from the completed three-seed paper package. They are small report tables, not raw datasets or trained checkpoints. Repository preparation did not regenerate their values.

## Included files

- `three_seed_all_results.csv`: 60 rows, one per main configuration and seed. Seeds are 42, 123, and 2026, with 20 configurations each.
- `three_seed_summary.csv`: 20 rows, one per configuration, with across-seed mean and sample standard deviation.
- `derived_statistics.csv`: reductions, accuracy differences, parameter counts, and other statistics computed for the research report.

## Column meanings

| Recorded column | Meaning |
| --- | --- |
| `seed` | Independent experiment seed |
| `run_key` | Configuration identifier, such as `frozen\|sigma1.0\|8bit` |
| `privacy_display` | Human-readable privacy condition; No-DP has no accounted epsilon |
| `method_display`, `bits` | Training regime and message precision |
| `learning_rate` | Validation-selected learning rate |
| `accuracy`, `macro_f1`, `weighted_f1` | Final test metrics, in percent |
| `epsilon_max`, `epsilon_mean`, `epsilon_min` | DP-SGD training ledger summaries at delta = 0.00001 |
| `training_upload_mb`, `training_download_mb`, `training_total_mb` | Serialized training model traffic in decimal MB |
| `runtime_minutes` | Recorded configuration runtime |

In the summary file, metric names gain `_mean` and `_sd`. The nested epsilon names distinguish the kind of client summary from the across-seed aggregation: `epsilon_max_mean`, for example, is the average of the three seeds' maximum client epsilon values.

No-DP epsilon cells are empty. An empty cell means not applicable, not epsilon zero and not perfect privacy. `derived_statistics.csv` is a two-column table of `statistic,value`.

## Relationship to newly generated combiner outputs

The combiner writes `all_seed_results.csv` and `three_seed_mean_std.csv` using a different column naming convention:

| Included report table | Combiner output |
| --- | --- |
| `accuracy` | `test_accuracy_percent` |
| `macro_f1` | `macro_f1_percent` |
| `weighted_f1` | `weighted_f1_percent` |
| `method_display` | `method` with lowercase values |
| `privacy_display` | `privacy_label`, such as `NoDP` or `sigma1.0` |
| Summary suffix `_sd` | Summary suffix `_std` |

The native combiner also writes per-client privacy rows, plots, a manifest, and min/max statistics. Those are generated from the three full result JSON files and are not reconstructed from these compact CSV tables.
