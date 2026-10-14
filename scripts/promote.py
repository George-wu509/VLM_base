"""Gate a candidate on the SAME holdout set, then create a local versioned release."""
import argparse
import json
import math
from pathlib import Path
from scripts.export import export
from evaluation.compare_checkpoints import compare
from training.config import read_yaml
from training.experiment import sha256

def check_gates(baseline, candidate, rules):
    failures = []
    for metric, rule in rules["metrics"].items():
        if metric not in baseline["metrics"] or metric not in candidate["metrics"]:
            failures.append(f"Missing metric {metric}")
            continue
        old, new = baseline["metrics"][metric], candidate["metrics"][metric]
        if not math.isfinite(old) or not math.isfinite(new):
            failures.append(f"{metric}: nonfinite value")
            continue
        direction = rule["direction"]
        if direction not in ("higher", "lower"):
            raise ValueError("Gate direction must be higher or lower")
        regression = old-new if direction == "higher" else new-old
        if regression > rule.get("max_regression", 0):
            failures.append(f"{metric}: regression {regression:.6f}")
        if "minimum" in rule and new < rule["minimum"]:
            failures.append(f"{metric}: below minimum {rule['minimum']}")
        if "maximum" in rule and new > rule["maximum"]:
            failures.append(f"{metric}: above maximum {rule['maximum']}")
    return failures

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rules", default="configs/promotion.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare([args.baseline_report, args.candidate_report])
    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_report).read_text(encoding="utf-8"))
    if candidate["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Candidate report does not describe the supplied checkpoint")
    rules = read_yaml(args.rules)
    failures = check_gates(baseline, candidate, rules)
    if failures:
        raise SystemExit("Promotion rejected:\n" + "\n".join(failures))
    model_path = export(args.checkpoint, args.output)
    (Path(args.output)/"release.json").write_text(json.dumps({"model": model_path, "rules": rules,
        "baseline_report": baseline, "candidate_report": candidate}, indent=2), encoding="utf-8")
    print(model_path)

if __name__ == "__main__":
    main()
