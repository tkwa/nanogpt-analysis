"""Analyze PR dates vs listed dates in the README."""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_pr_info(pr_number: int, repo: str = "KellerJordan/modded-nanogpt") -> dict | None:
    """Get PR information from GitHub using gh CLI."""
    try:
        result = subprocess.run(
            [
                "gh", "pr", "view", str(pr_number),
                "--repo", repo,
                "--json", "number,title,createdAt,mergedAt,closedAt",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"  Failed to get PR #{pr_number}: {result.stderr}")
            return None

        import json as json_module
        return json_module.loads(result.stdout)

    except subprocess.TimeoutExpired:
        print(f"  Timeout getting PR #{pr_number}")
        return None
    except FileNotFoundError:
        print("  gh CLI not found. Please install GitHub CLI.")
        return None


def analyze_pr_dates(records_path: str) -> list[dict]:
    """Analyze PR dates vs listed dates."""
    with open(records_path, "r") as f:
        records = json.load(f)

    # Filter records with PR numbers
    records_with_pr = [r for r in records if r["pr_number"] is not None and not r["is_retiming"]]

    print(f"Found {len(records_with_pr)} records with PR numbers")

    results = []

    for record in records_with_pr:
        pr_number = record["pr_number"]
        listed_date = record["date"]
        record_num = record["record_num"]

        print(f"Fetching PR #{pr_number} (record #{record_num})...")

        pr_info = get_pr_info(pr_number)

        if pr_info is None:
            results.append({
                "record_num": record_num,
                "pr_number": pr_number,
                "listed_date": listed_date,
                "pr_created": None,
                "pr_merged": None,
                "error": "Failed to fetch",
            })
            continue

        # Parse dates
        pr_created = pr_info.get("createdAt", "")[:10] if pr_info.get("createdAt") else None
        pr_merged = pr_info.get("mergedAt", "")[:10] if pr_info.get("mergedAt") else None

        # Calculate differences
        created_diff = None
        merged_diff = None

        if listed_date and pr_created:
            listed_dt = datetime.strptime(listed_date, "%Y-%m-%d")
            created_dt = datetime.strptime(pr_created, "%Y-%m-%d")
            created_diff = (listed_dt - created_dt).days

        if listed_date and pr_merged:
            listed_dt = datetime.strptime(listed_date, "%Y-%m-%d")
            merged_dt = datetime.strptime(pr_merged, "%Y-%m-%d")
            merged_diff = (listed_dt - merged_dt).days

        results.append({
            "record_num": record_num,
            "pr_number": pr_number,
            "listed_date": listed_date,
            "pr_created": pr_created,
            "pr_merged": pr_merged,
            "days_listed_vs_created": created_diff,
            "days_listed_vs_merged": merged_diff,
        })

    return results


def main():
    data_dir = Path(__file__).parent.parent / "data"
    records_path = data_dir / "raw_records.json"
    output_path = data_dir / "pr_date_analysis.json"

    results = analyze_pr_dates(str(records_path))

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved PR date analysis to {output_path}")

    # Summary statistics
    valid = [r for r in results if r.get("days_listed_vs_merged") is not None]

    if valid:
        diffs = [r["days_listed_vs_merged"] for r in valid]
        created_diffs = [r["days_listed_vs_created"] for r in valid if r.get("days_listed_vs_created") is not None]

        print("\n" + "=" * 60)
        print("PR DATE ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nRecords analyzed: {len(valid)}")

        print("\nListed Date vs PR Merged Date:")
        print(f"  Mean difference: {sum(diffs)/len(diffs):.1f} days")
        print(f"  Min: {min(diffs)} days")
        print(f"  Max: {max(diffs)} days")

        if created_diffs:
            print("\nListed Date vs PR Created Date:")
            print(f"  Mean difference: {sum(created_diffs)/len(created_diffs):.1f} days")
            print(f"  Min: {min(created_diffs)} days")
            print(f"  Max: {max(created_diffs)} days")

        # Find notable discrepancies
        print("\nNotable discrepancies (>3 days difference):")
        notable = [r for r in valid if abs(r["days_listed_vs_merged"]) > 3]
        if notable:
            for r in notable:
                print(f"  Record #{r['record_num']} (PR #{r['pr_number']}): "
                      f"listed={r['listed_date']}, merged={r['pr_merged']}, "
                      f"diff={r['days_listed_vs_merged']} days")
        else:
            print("  None found")

    else:
        print("\nNo valid PR date data to analyze")


if __name__ == "__main__":
    main()
