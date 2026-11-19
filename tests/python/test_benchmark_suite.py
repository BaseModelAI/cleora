"""The parts of the benchmark suite that decide what the result tables say: which
variant of a method is reported, how methods are ranked, which iteration counts are
"tuned". They only need NumPy, so they are tested here on fabricated job files."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))

from cleora_bench import configurations, iteration_fit, report, selection  # noqa: E402


def write_job(results, dataset, method, dim, seed, task, validation, test, metric="accuracy", probe="logistic"):
    if task == "nc":
        probes = {probe: {"metrics": [{metric: test}], "chosen": [{"validation_score": validation}]}}
    else:
        probes = {probe: {"metrics": {metric: test}, "chosen": {"validation_mrr": validation}}}
    job = {"dataset": dataset, "method": method, "dim": dim, "seed": seed, "task": task, "probes": probes,
           "embedding": {"status": "ok", "seconds": 1.0, "peak_rss_mb": 10.0, "nodes": 100, "edges": 400}}
    path = pathlib.Path(results) / dataset / method / f"dim{dim}" / f"seed{seed}" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job))


def test_the_variant_is_chosen_on_validation_and_its_test_score_reported(tmp_path):
    # 128 dimensions validate better; 1024 happens to test better
    for seed, (v128, t128, v1024, t1024) in enumerate([(0.80, 0.70, 0.75, 0.90), (0.82, 0.72, 0.77, 0.92)]):
        write_job(tmp_path, "toy", "deepwalk", 128, seed, "nc", v128, t128)
        write_job(tmp_path, "toy", "deepwalk", 1024, seed, "nc", v1024, t1024)
    table = report.aggregate(report.load_jobs(str(tmp_path)))
    cell = report.cell(table, "toy", "nc", "logistic", "accuracy", [("deepwalk", 1024), ("deepwalk", 128)])
    assert cell["variant"] == ["deepwalk", 128]
    assert cell["mean"] == pytest.approx(0.71) and cell["runs"] == 2
    assert cell["std"] == pytest.approx(0.01)


def test_failed_and_missing_runs_are_told_apart(tmp_path):
    job = {"dataset": "toy", "method": "netmf", "dim": 1024, "seed": 0, "task": "nc",
           "embedding": {"status": "skipped", "reason": "needs dense n x n matrices; limited to 60000 nodes"}}
    path = tmp_path / "toy" / "netmf" / "dim1024" / "seed0" / "nc.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(job))
    table = report.aggregate(report.load_jobs(str(tmp_path)))
    skipped = report.cell(table, "toy", "nc", "logistic", "accuracy", [("netmf", 1024)])
    assert report.format_cell(skipped, None, None) == "n/a (dense)"
    never_run = report.cell(table, "toy", "nc", "logistic", "accuracy", [("hope", 1024)])
    assert report.format_cell(never_run, None, None) == "–"


def test_ranks_share_ties_and_put_missing_results_last():
    cells = {"a": {"mean": 0.9}, "b": {"mean": 0.8}, "c": {"mean": 0.9}, "d": {"missing": True}}
    assert report.ranks(cells) == {"a": 1.5, "b": 3.0, "c": 1.5, "d": 4.0}
    assert report.ranks(cells, higher_is_better=False)["b"] == 1.0


def test_summary_counts_wins_and_averages_ranks():
    rows = {"g1": {"a": {"mean": 0.9}, "b": {"mean": 0.8}}, "g2": {"a": {"mean": 0.5}, "b": {"mean": 0.7}},
            "g3": {"a": {"mean": 0.6}, "b": {"missing": True}}, "empty": {"a": {"missing": True}, "b": {"missing": True}}}
    summary = report.summary_rows(rows, ["a", "b"])
    assert summary["a"] == {"mean_rank": pytest.approx(4 / 3), "wins": 2, "results": 3, "rows": 3}
    assert summary["b"] == {"mean_rank": pytest.approx(5 / 3), "wins": 1, "results": 2, "rows": 3}


def test_tuned_iterations_are_the_ones_that_validate_best(tmp_path):
    sweep = tmp_path / selection.SWEEP_DIRECTORY
    for iterations, name in zip(configurations.SWEEP_ITERATIONS, configurations.sweep("hyperedge", "left")):
        # validation peaks at 8 iterations, test at 16
        write_job(sweep, "toy", name, 1024, 0, "nc", 1.0 - abs(iterations - 8) / 100, 1.0 - abs(iterations - 16) / 100)
        write_job(sweep, "toy", name, 1024, 0, "lp", 1.0 - abs(iterations - 3) / 100, 0.5, metric="mrr", probe="cosine")
    chosen = selection.write(str(tmp_path))
    assert chosen == {"toy": {"nc": {"hyperedge/left": "cleora_left_k8"}, "lp": {"hyperedge/left": "cleora_left_k3"}}}
    assert selection.configurations_for(str(tmp_path), "toy", "nc") == ["cleora_left_k8"]
    assert selection.configurations_for(str(tmp_path), "other", "nc") == []


def test_configuration_names():
    assert len(configurations.sweep("hyperedge")) == len(configurations.sweep("pair")) == 2 * len(configurations.SWEEP_ITERATIONS)
    assert configurations.automatic("classification") == [
        "cleora_left_cls", "cleora_sym_cls", "cleora_pair_left_cls", "cleora_pair_sym_cls"]
    method, parameters = configurations.resolve("cleora_pair_sym_sim")
    assert method == "cleora_classic"
    assert parameters == {"iterations": "auto", "purpose": "similarity", "propagation": "symmetric", "columns": "src dst"}
    with pytest.raises(KeyError, match="Unknown configuration"):
        configurations.resolve("cleora_best")


def test_regret_is_measured_in_points_against_the_validated_optimum(tmp_path):
    for iterations, name in zip(configurations.SWEEP_ITERATIONS, configurations.sweep("pair", "left")):
        write_job(tmp_path, "toy", name, 1024, 0, "nc", 1.0 - abs(iterations - 8) / 100, 0.5 + iterations / 1000)
    (record,) = iteration_fit.curves(str(tmp_path), "nc")
    assert record["best_iterations"] == 8 and record["degree"] == 8.0 and record["share"] == 0.0
    assert iteration_fit.regret(record, 8) == pytest.approx(0.0)
    assert iteration_fit.regret(record, 4) == pytest.approx(0.4)   # 0.508 - 0.504, in points
    # between grid points: linear in log K (halfway between 16 and 24 on that scale is 19.6)
    assert iteration_fit.regret(record, np.sqrt(16 * 24)) == pytest.approx(-1.2)
