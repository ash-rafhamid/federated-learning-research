"""Check packaged source/notebooks and recorded tables without running training."""

import ast
import copy
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = {42, 123, 2026}
GENERIC = "Pipeline_1_Final_Corrected_One_Seed_Kaggle.py"
COMBINER = "Pipeline_1_Final_Combine_Three_Seeds_Kaggle.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_tree(source, name):
    return ast.parse(source, filename=name)


def normalized_tree(tree):
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUN_SEED"
            for target in node.targets
        ):
            node.value = ast.Constant(value=42)
    return ast.dump(tree, include_attributes=False)


def check_notebook(path, expected, expected_seed=None):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    require(notebook.get("nbformat") == 4, f"Invalid notebook version: {path.name}")
    cells = notebook["cells"]
    require(bool(cells), f"Empty notebook: {path.name}")
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    require(bool(code_cells), f"No code cells: {path.name}")
    require(
        all(not cell.get("outputs") for cell in code_cells),
        f"Unexpected saved notebook output: {path.name}",
    )
    source = "\n\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in code_cells
    )
    tree = source_tree(source, path.name)
    require(normalized_tree(tree) == expected, f"Script mismatch: {path.name}")
    if expected_seed is not None:
        values = [
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "RUN_SEED"
                for target in node.targets
            )
        ]
        require(values == [expected_seed], f"Wrong seed in {path.name}: {values}")


def check_results():
    with (ROOT / "results/three_seed_all_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 60, f"Expected 60 rows, found {len(rows)}")
    counts = Counter(int(row["seed"]) for row in rows)
    require(counts == Counter({seed: 20 for seed in SEEDS}), f"Wrong seed counts: {counts}")
    expected_keys = {
        f"{method}|{privacy}|{bits}bit"
        for privacy in ("NoDP", "sigma0.5", "sigma1.0", "sigma2.0")
        for method, widths in (("baseline", [32]), ("pretrained", [32]), ("frozen", [32, 8, 4]))
        for bits in widths
    }
    for seed in SEEDS:
        keys = {row["run_key"] for row in rows if int(row["seed"]) == seed}
        require(keys == expected_keys, f"Incomplete configuration matrix for seed {seed}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["run_key"]].append(row)
        total = float(row["training_total_mb"])
        parts = float(row["training_upload_mb"]) + float(row["training_download_mb"])
        require(math.isclose(total, parts, rel_tol=1e-9), f"Byte sum mismatch: {row['run_key']}")
    with (ROOT / "results/three_seed_summary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        summaries = list(csv.DictReader(handle))
    require(len(summaries) == 20, f"Expected 20 summary rows, found {len(summaries)}")
    require(
        {row["run_key"] for row in summaries} == expected_keys,
        "Summary configuration matrix mismatch",
    )
    fields = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "epsilon_max": "epsilon_max",
        "epsilon_mean": "epsilon_mean",
        "epsilon_min": "epsilon_min",
        "training_total_mb": "training_total_mb",
        "runtime_minutes": "runtime_minutes",
    }
    for summary in summaries:
        for raw_field, summary_field in fields.items():
            values = [
                float(row[raw_field]) for row in groups[summary["run_key"]]
                if row[raw_field] != ""
            ]
            for suffix, statistic in (("mean", statistics.mean), ("sd", statistics.stdev)):
                recorded = summary[f"{summary_field}_{suffix}"]
                if not values:
                    require(recorded == "", f"Unexpected No-DP epsilon: {summary['run_key']}")
                    continue
                require(
                    math.isclose(float(recorded), statistic(values), rel_tol=1e-8, abs_tol=1e-8),
                    f"Summary mismatch: {summary['run_key']} / {summary_field}_{suffix}",
                )
    print("PASS: 60 result rows, 20 complete configurations per seed, and recomputed summary statistics")


def main():
    generic = source_tree((ROOT / GENERIC).read_text(encoding="utf-8-sig"), GENERIC)
    combiner = source_tree((ROOT / COMBINER).read_text(encoding="utf-8-sig"), COMBINER)
    for seed in sorted(SEEDS):
        check_notebook(
            ROOT / f"Pipeline_1_Final_Seed_{seed}_Kaggle.ipynb",
            normalized_tree(generic),
            seed,
        )
    check_notebook(
        ROOT / "Pipeline_1_Final_Combine_Three_Seeds_Kaggle.ipynb",
        normalized_tree(combiner),
    )
    print("PASS: scripts parse; all four notebook programs match their source and seed settings")
    check_results()
    print("Repository checks passed. Training was not executed.")


if __name__ == "__main__":
    main()
