"""Extract timing records from the modded-nanogpt README."""

import re
import json
from datetime import datetime
from pathlib import Path


def parse_time_to_minutes(time_str: str) -> float:
    """Convert time string to minutes (float)."""
    time_str = time_str.strip().lower()

    # Handle "X hours" format
    if "hour" in time_str:
        match = re.search(r"([\d.]+)\s*hour", time_str)
        if match:
            return float(match.group(1)) * 60

    # Handle "X minutes" format
    if "minute" in time_str:
        match = re.search(r"([\d.]+)\s*minute", time_str)
        if match:
            return float(match.group(1))

    # Handle plain number (assumed to be minutes)
    try:
        return float(time_str)
    except ValueError:
        return None


def parse_date(date_str: str) -> str:
    """Parse MM/DD/YY date format to ISO format."""
    date_str = date_str.strip()
    if not date_str or date_str == "-":
        return None

    try:
        dt = datetime.strptime(date_str, "%m/%d/%y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def is_retiming(contributors: str, description: str) -> bool:
    """Check if this record is just a retiming, not an actual improvement."""
    text = (contributors + " " + description).lower()
    retiming_indicators = [
        "not a new record",
        "just re-timing",
        "just retiming",
    ]
    return any(indicator in text for indicator in retiming_indicators)


def extract_pr_number(line: str) -> int | None:
    """Extract PR number from table row if present."""
    # Look for [PR](https://github.com/.../pull/XXX) anywhere in the line
    match = re.search(r"\[PR\]\([^)]*pull/(\d+)\)", line)
    if match:
        return int(match.group(1))
    return None


def extract_records_from_readme(readme_path: str) -> list[dict]:
    """Extract all records from Track 1 table in README."""
    with open(readme_path, "r") as f:
        content = f.read()

    # Find Track 1 table (GPT-2 Small, target ≤3.28)
    # The table starts after "## World record history" and ends before "## Rules"
    track1_match = re.search(
        r"## World record history.*?\| # \| Record time.*?\n\| - \| - \|.*?\n(.*?)(?=\n## Rules|\n---\n### Timing change)",
        content,
        re.DOTALL
    )

    if not track1_match:
        raise ValueError("Could not find Track 1 table in README")

    table_content = track1_match.group(1)
    records = []

    for line in table_content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("| -"):
            continue

        # Parse table row: | # | Record time | Description | Date | Log | Contributors |
        # Note: The actual format doesn't have leading pipe in the README
        parts = [p.strip() for p in line.split("|")]

        # Filter empty parts
        parts = [p for p in parts if p]

        if len(parts) < 6:
            continue

        record_num, time_str, description, date_str, log, contributors = parts[:6]

        # Skip header-like rows
        if record_num == "#" or record_num == "-":
            continue

        try:
            record_num = int(record_num)
        except ValueError:
            continue

        time_minutes = parse_time_to_minutes(time_str)
        date_iso = parse_date(date_str)
        pr_number = extract_pr_number(line)

        # Clean description - remove markdown links
        clean_desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", description)

        record = {
            "record_num": record_num,
            "time_minutes": time_minutes,
            "description": clean_desc,
            "date": date_iso,
            "pr_number": pr_number,
            "contributors": contributors,
            "is_retiming": is_retiming(contributors, description),
        }

        records.append(record)

    return records


def main():
    repo_path = Path(__file__).parent.parent / "data" / "modded-nanogpt"
    readme_path = repo_path / "README.md"
    output_path = Path(__file__).parent.parent / "data" / "raw_records.json"

    records = extract_records_from_readme(readme_path)

    print(f"Extracted {len(records)} total records")
    print(f"  - Actual improvements: {sum(1 for r in records if not r['is_retiming'])}")
    print(f"  - Retimings: {sum(1 for r in records if r['is_retiming'])}")

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Print summary
    print("\nFirst 5 records:")
    for r in records[:5]:
        print(f"  #{r['record_num']}: {r['time_minutes']:.2f} min on {r['date']}")

    print("\nLast 5 records:")
    for r in records[-5:]:
        status = " (retiming)" if r["is_retiming"] else ""
        print(f"  #{r['record_num']}: {r['time_minutes']:.3f} min on {r['date']}{status}")


if __name__ == "__main__":
    main()
