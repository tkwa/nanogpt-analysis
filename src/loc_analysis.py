"""Analyze lines of code in train_gpt.py over time."""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def run_git(args: list[str], repo_path: str) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_commit_at_date(repo_path: str, target_date: str) -> str | None:
    """Find the most recent commit on or before target_date that modified train_gpt.py."""
    # Get all commits that modified train_gpt.py before the target date
    result = run_git(
        [
            "log",
            "--format=%H",
            f"--until={target_date}T23:59:59",
            "--",
            "train_gpt.py",
        ],
        repo_path,
    )

    if result:
        # Return the most recent commit (first line)
        return result.split("\n")[0]

    # If no commits modified train_gpt.py by that date, try to find any commit
    result = run_git(
        ["log", "--format=%H", f"--until={target_date}T23:59:59", "-1"],
        repo_path,
    )

    return result if result else None


def get_loc_at_commit(repo_path: str, commit: str, filename: str = "train_gpt.py") -> tuple[int | None, str | None]:
    """Get lines of code in a file at a specific commit.

    Returns (loc, actual_filename) tuple.
    """
    result = run_git(["show", f"{commit}:{filename}"], repo_path)
    if result:
        return len(result.split("\n")), filename

    # Try train_gpt2.py as fallback (old name)
    if filename == "train_gpt.py":
        result = run_git(["show", f"{commit}:train_gpt2.py"], repo_path)
        if result:
            return len(result.split("\n")), "train_gpt2.py"

    return None, None


def get_all_training_files_at_commit(repo_path: str, commit: str) -> list[str]:
    """List all .py files at a commit to identify potential training files."""
    result = run_git(["ls-tree", "-r", "--name-only", commit], repo_path)
    if result:
        files = result.split("\n")
        py_files = [f for f in files if f.endswith(".py")]
        return py_files
    return []


def analyze_loc_for_records(records_path: str, repo_path: str) -> list[dict]:
    """Analyze LoC for each record."""
    with open(records_path, "r") as f:
        records = json.load(f)

    results = []

    for record in records:
        if record["is_retiming"]:
            continue

        date = record["date"]
        record_num = record["record_num"]

        if not date:
            print(f"  Record #{record_num}: No date available")
            results.append({
                "record_num": record_num,
                "date": None,
                "commit": None,
                "loc_train_gpt": None,
            })
            continue

        commit = get_commit_at_date(repo_path, date)

        if not commit:
            print(f"  Record #{record_num} ({date}): No commit found")
            results.append({
                "record_num": record_num,
                "date": date,
                "commit": None,
                "loc_train_gpt": None,
            })
            continue

        loc, actual_filename = get_loc_at_commit(repo_path, commit, "train_gpt.py")
        commit_short = commit[:7]

        if loc:
            print(f"  Record #{record_num} ({date}): {loc} LoC @ {commit_short} ({actual_filename})")
        else:
            print(f"  Record #{record_num} ({date}): training file not found @ {commit_short}")

        results.append({
            "record_num": record_num,
            "date": date,
            "commit": commit_short,
            "loc_train_gpt": loc,
            "training_file": actual_filename,
        })

    return results


def main():
    repo_path = Path(__file__).parent.parent / "data" / "modded-nanogpt"
    records_path = Path(__file__).parent.parent / "data" / "raw_records.json"
    output_path = Path(__file__).parent.parent / "data" / "loc_data.json"

    print("Analyzing lines of code for each record...")
    results = analyze_loc_for_records(str(records_path), str(repo_path))

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Summary
    valid_results = [r for r in results if r["loc_train_gpt"] is not None]
    print(f"\nSummary: {len(valid_results)}/{len(results)} records have LoC data")

    if valid_results:
        print(f"  First LoC: {valid_results[0]['loc_train_gpt']} (record #{valid_results[0]['record_num']})")
        print(f"  Last LoC: {valid_results[-1]['loc_train_gpt']} (record #{valid_results[-1]['record_num']})")


if __name__ == "__main__":
    main()
