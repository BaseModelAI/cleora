"""The parts of the benchmark suite that decide what the result tables say: which
variant of a method is reported, how methods are ranked, which iteration counts are
"tuned". They only need NumPy, so they are tested here on fabricated job files."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))

from cleora_bench import configurations, iteration_fit, popularity, report, selection  # noqa: E402


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
           "embedding": {"status": "skipped", "reason": "this implementation is limited to 60000 nodes"}}
    path = tmp_path / "toy" / "netmf" / "dim1024" / "seed0" / "nc.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(job))
    table = report.aggregate(report.load_jobs(str(tmp_path)))
    skipped = report.cell(table, "toy", "nc", "logistic", "accuracy", [("netmf", 1024)])
    assert report.format_cell(skipped, None, None) == "n/a (size)"
    never_run = report.cell(table, "toy", "nc", "logistic", "accuracy", [("hope", 1024)])
    assert report.format_cell(never_run, None, None) == "–"


def test_ranks_share_ties_and_put_missing_results_last():
    cells = {"a": {"mean": 0.9}, "b": {"mean": 0.8}, "c": {"mean": 0.9}, "d": {"missing": True}}
    assert report.ranks(cells) == {"a": 1.5, "b": 3.0, "c": 1.5, "d": 4.0}
    assert report.ranks(cells, higher_is_better=False)["b"] == 1.0


def test_summary_counts_wins_and_averages_ranks():
    rows = {"g1": {"a": {"mean": 0.9}, "b": {"mean": 0.8}, "c": {"mean": 0.7}},
            "g2": {"a": {"mean": 0.5}, "b": {"mean": 0.7}, "c": {"missing": True}},
            # being the only method with a result is not a win: the row is not compared
            "alone": {"a": {"mean": 0.6}, "b": {"missing": True}, "c": {"missing": True}},
            "empty": {"a": {"missing": True}, "b": {"missing": True}, "c": {"missing": True}}}
    summary = report.summary_rows(rows, ["a", "b", "c"])
    assert summary["a"] == {"mean_rank": pytest.approx(1.5), "wins": 1, "results": 2, "rows": 2}
    assert summary["b"] == {"mean_rank": pytest.approx(1.5), "wins": 1, "results": 2, "rows": 2}
    assert summary["c"] == {"mean_rank": pytest.approx(3.0), "wins": 0, "results": 1, "rows": 2}


def test_tuned_iterations_are_the_ones_that_validate_best(tmp_path):
    sweep = tmp_path / selection.SWEEP_DIRECTORY
    for iterations, name in zip(configurations.SWEEP_ITERATIONS, configurations.sweep("hyperedge", "left")):
        # validation peaks at 8 iterations, test at 16
        write_job(sweep, "toy", name, 1024, 0, "nc", 1.0 - abs(iterations - 8) / 100, 1.0 - abs(iterations - 16) / 100)
        write_job(sweep, "toy", name, 1024, 0, "lp", 1.0 - abs(iterations - 3) / 100, 0.5, metric="mrr", probe="cosine")
    chosen = selection.write(str(tmp_path))
    # the count chosen for the link task is run as the similarity configuration
    assert chosen == {"toy": {"nc": {"hyperedge/left": "cleora_left_k8"}, "lp": {"hyperedge/left": "cleora_left_k3_sim"}}}
    assert configurations.iterations("cleora_left_k3_sim") == 3
    assert configurations.resolve("cleora_left_k3_sim")[1]["popularity"] == configurations.SIMILARITY["popularity"]
    assert "popularity" not in configurations.resolve("cleora_left_k8")[1]
    assert selection.configurations_for(str(tmp_path), "toy", "nc") == ["cleora_left_k8"]
    assert selection.configurations_for(str(tmp_path), "other", "nc") == []


def test_configuration_names():
    assert len(configurations.sweep("hyperedge")) == len(configurations.sweep("pair")) == 2 * len(configurations.SWEEP_ITERATIONS)
    assert configurations.automatic("classification") == [
        "cleora_left_cls", "cleora_sym_cls", "cleora_pair_left_cls", "cleora_pair_sym_cls"]
    method, parameters = configurations.resolve("cleora_pair_sym_sim")
    assert method == "cleora_classic"
    assert parameters == {"iterations": "auto", "purpose": "similarity", "propagation": "symmetric", "columns": "src dst",
                          **configurations.SIMILARITY}
    assert "popularity" not in configurations.resolve("cleora_pair_sym_cls")[1]
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


def test_popularity_table_follows_the_rule_of_the_benchmark(tmp_path):
    # one graph that the prior helps (by validation MRR) and one that it hurts
    helped = {str(e): {"mrr": 0.30 + (0.1 if e == 0.0625 else 0.05 if e > 0 else 0), "recall@10": 0.20 + e} for e in popularity.EXPONENTS}
    hurt = {str(e): {"mrr": 0.50 - e, "recall@10": 0.40 - e} for e in popularity.EXPONENTS}
    for name, scores in (("helped", helped), ("hurt", hurt)):
        target = pathlib.Path(popularity.path(str(tmp_path), name))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(scores))
    rows = {line.split("|")[1].strip(): [cell.strip().strip("*") for cell in line.split("|")[2:-1]]
            for line in popularity.markdown(str(tmp_path / popularity.DIRECTORY)).splitlines()[2:]}
    # applied everywhere the prior is averaged over both graphs; under the rule the second graph keeps the cosine
    assert rows["0.0625"] == ["2", "0.4188", "0.3000", "0.4500", "0.3313", "1"]
    assert rows["0"] == ["2", "0.4000", "0.3000", "0.4000", "0.3000", "0"]
    assert popularity.markdown(str(tmp_path / "nothing")) == ""
