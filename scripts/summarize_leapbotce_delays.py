import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    baseline = None
    for delay_dir in sorted(args.root.glob("delay_*"), key=lambda path: int(path.name.split("_")[-1])):
        files = list(delay_dir.glob("**/*_results.json"))
        successes = total = 0
        for path in files:
            result = json.loads(path.read_text())
            successes += int(result["successes"])
            total += int(result.get("total_episodes", len(result.get("success_episodes", [])) + len(result.get("failure_episodes", []))))
        rate = successes / total if total else 0.0
        if baseline is None:
            baseline = rate
        retention = rate / baseline if baseline and baseline > 0 else 0.0
        rows.append((delay_dir.name.split("_")[-1], successes, total, rate, retention))
    print("delay\tsuccess\ttotal\trate\tretention")
    for delay, successes, total, rate, retention in rows:
        print(f"{delay}\t{successes}\t{total}\t{rate:.4f}\t{retention:.4f}")


if __name__ == "__main__":
    main()
