import json
from pathlib import Path

from utils.report_generator import generate_run_report


def test_generate_run_report_surfaces_directional_pair_metrics(tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "report"
    run_dir.mkdir()
    out_dir.mkdir()

    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "train_config.json").write_text(json.dumps({"active_finger_head": True}))
    (out_dir / "eval_manifest.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "action_acc": 0.9,
                    "joint_acc": 0.8,
                    "finger_acc_non_rest": 0.85,
                    "applicability_fp_rate_on_true_rest": 0.1,
                    "applicability_fn_rate_on_true_non_rest": 0.05,
                    "action_applicability_disagreement_rate": 0.0667,
                    "raw_non_rest_none_count": 3,
                    "raw_non_rest_none_rate": 0.1,
                    "raw_rest_non_none_count": 1,
                    "raw_rest_non_none_rate": 0.0333,
                    "committed_non_rest_none_count": 0,
                    "committed_non_rest_none_rate": 0.0,
                    "committed_rest_non_none_count": 0,
                    "committed_rest_non_none_rate": 0.0,
                    "deployment_pair_invariant_ok": True,
                    "raw_valid_pair_rate": 0.8667,
                    "raw_invalid_pair_rate": 0.1333,
                },
                "benchmarks": {
                    "primary_mixed_holdout": {
                        "test_n": 30,
                        "test_action_counts": {"REST": 10, "OPEN": 10, "CLOSE": 10},
                        "test_finger_counts": {"NONE": 10, "THUMB": 10, "INDEX": 10},
                        "metrics": {
                            "action_acc": 0.9,
                            "joint_acc": 0.8,
                            "finger_acc_non_rest": 0.85,
                            "applicability_fp_rate_on_true_rest": 0.1,
                            "applicability_fn_rate_on_true_non_rest": 0.05,
                            "action_applicability_disagreement_rate": 0.0667,
                            "raw_non_rest_none_rate": 0.1,
                            "raw_rest_non_none_rate": 0.0333,
                            "committed_rest_non_none_rate": 0.0,
                            "committed_non_rest_none_rate": 0.0,
                            "deployment_pair_invariant_ok": True,
                        },
                    }
                },
            }
        )
    )

    report_path = generate_run_report(run_dir, out_dir=out_dir)
    html = report_path.read_text()

    assert "Raw non-REST+NONE count/rate" in html
    assert "Committed REST+active count/rate" in html
    assert "Committed non-REST+NONE count/rate" in html
    assert "Applicability FP(rest) / FN(non-REST)" in html
    assert "Action/applicability disagreement" in html
    assert "Deployment pair invariant OK" in html
    assert "Deprecated raw valid/invalid pair rate" in html
