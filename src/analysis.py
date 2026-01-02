"""Compile all data into improvements.json."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_star_history(star_csv_path: str | None) -> pd.DataFrame | None:
    """Load star history from CSV if available.

    Expected format from star-history.com:
    - Column 1: Date (YYYY-MM-DD)
    - Column 2: Stars (integer)
    """
    if star_csv_path is None or not Path(star_csv_path).exists():
        return None

    df = pd.read_csv(star_csv_path)

    # Standardize column names
    df.columns = ["date", "stars"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df


def interpolate_stars(star_df: pd.DataFrame | None, target_date: str) -> int | None:
    """Linearly interpolate stars at a specific date."""
    if star_df is None or target_date is None:
        return None

    target_dt = pd.to_datetime(target_date)

    # Find the closest dates before and after
    before = star_df[star_df["date"] <= target_dt]
    after = star_df[star_df["date"] >= target_dt]

    if before.empty:
        # Target date is before all star data
        return None

    if after.empty:
        # Target date is after all star data, use last known value
        return int(before.iloc[-1]["stars"])

    if target_dt in star_df["date"].values:
        # Exact match
        return int(star_df[star_df["date"] == target_dt].iloc[0]["stars"])

    # Linear interpolation
    before_row = before.iloc[-1]
    after_row = after.iloc[0]

    days_total = (after_row["date"] - before_row["date"]).days
    days_from_start = (target_dt - before_row["date"]).days

    if days_total == 0:
        return int(before_row["stars"])

    stars_diff = after_row["stars"] - before_row["stars"]
    interpolated = before_row["stars"] + (stars_diff * days_from_start / days_total)

    return int(round(interpolated))


def compile_improvements(
    records_path: str,
    loc_path: str,
    star_csv_path: str | None = None,
) -> list[dict]:
    """Compile all data sources into improvements list."""
    # Load records
    with open(records_path, "r") as f:
        records = json.load(f)

    # Load LoC data
    with open(loc_path, "r") as f:
        loc_data = json.load(f)

    # Create LoC lookup
    loc_lookup = {r["record_num"]: r for r in loc_data}

    # Load star history if available
    star_df = load_star_history(star_csv_path)

    if star_df is not None:
        print(f"Loaded star history: {len(star_df)} data points")
        print(f"  Date range: {star_df['date'].min()} to {star_df['date'].max()}")
    else:
        print("No star history available (will leave stars column empty)")

    # Find start date (first record date)
    start_date = None
    for r in records:
        if r["date"] and not r["is_retiming"]:
            start_date = datetime.strptime(r["date"], "%Y-%m-%d")
            break

    improvements = []

    for record in records:
        if record["is_retiming"]:
            continue

        record_num = record["record_num"]
        date = record["date"]

        # Calculate days since start
        days_since_start = None
        if date and start_date:
            record_date = datetime.strptime(date, "%Y-%m-%d")
            days_since_start = (record_date - start_date).days

        # Get LoC
        loc_info = loc_lookup.get(record_num, {})
        cumulative_loc = loc_info.get("loc_train_gpt")

        # Interpolate stars
        stars = interpolate_stars(star_df, date)

        improvement = {
            "record_num": record_num,
            "date": date,
            "days_since_start": days_since_start,
            "record_time_minutes": record["time_minutes"],
            "cumulative_loc": cumulative_loc,
            "stars": stars,
            "description": record["description"],
            "pr_number": record["pr_number"],
        }

        improvements.append(improvement)

    return improvements


def main():
    data_dir = Path(__file__).parent.parent / "data"
    records_path = data_dir / "raw_records.json"
    loc_path = data_dir / "loc_data.json"
    star_csv_path = data_dir / "star_history.csv"
    output_path = data_dir / "improvements.json"

    # Check if star history exists
    star_path = str(star_csv_path) if star_csv_path.exists() else None

    improvements = compile_improvements(
        str(records_path),
        str(loc_path),
        star_path,
    )

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(improvements, f, indent=2)

    print(f"\nCompiled {len(improvements)} improvements to {output_path}")

    # Summary statistics
    with_loc = sum(1 for i in improvements if i["cumulative_loc"] is not None)
    with_stars = sum(1 for i in improvements if i["stars"] is not None)

    print(f"\nData coverage:")
    print(f"  - Records with LoC: {with_loc}/{len(improvements)}")
    print(f"  - Records with stars: {with_stars}/{len(improvements)}")

    # Print sample
    print("\nSample (first 3 records):")
    for imp in improvements[:3]:
        print(f"  #{imp['record_num']}: {imp['record_time_minutes']:.2f} min, "
              f"LoC={imp['cumulative_loc']}, stars={imp['stars']}, "
              f"days={imp['days_since_start']}")


if __name__ == "__main__":
    main()
