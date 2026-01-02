# NanoGPT Speedrun Analysis - Implementation Notes

## Overview

This project analyzes the algorithmic improvement in the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt) against various metrics:
- Calendar time (days since first record)
- Lines of code in training script
- Repository stars (when data is available)

## Data Sources

### 1. Timing Records (`data/raw_records.json`)
- Extracted from the modded-nanogpt README markdown table
- **Track 1 only** (GPT-2 Small, ≤3.28 val loss target)
- 55 actual improvements + 2 retimings excluded
- Dates in MM/DD/YY format, converted to ISO format

### 2. Lines of Code (`data/loc_data.json`)
- Counts lines in `train_gpt.py` (or `train_gpt2.py` for early records)
- Uses git history to find the commit closest to each record's date
- **54/55 records have LoC data** (record #1 is llm.c baseline, predates this repo)

### 3. Star History (`data/star_history.csv`)
- **User must manually provide** from https://www.star-history.com/
- Expected format: CSV with date and star count columns
- Stars are linearly interpolated to match record dates

## Implementation Details

### Record Extraction (`src/extract_records.py`)
- Parses markdown table using regex
- Identifies retimings by looking for "not a new record" or "just re-timing" in contributors/description
- Extracts PR numbers from `[PR](url)` links in the log column

### Lines of Code Analysis (`src/loc_analysis.py`)
- For each record date, finds the most recent commit that modified the training file
- Handles filename change: `train_gpt2.py` → `train_gpt.py` (around record #18)
- Only counts the main training file, not supporting scripts

### Analysis Compilation (`src/analysis.py`)
- Merges records, LoC, and star data
- Calculates `days_since_start` from first record date (2024-05-28)
- Outputs unified `improvements.json`

### Plotting (`src/plots.py`)
- Fits both power law and exponential models to time vs days
- Power law: `time = a * (days+1)^b`
- Exponential: `time = a * exp(b * days)`
- Outputs metrics to `data/metrics.yaml`

## Key Findings

### Regression Results
- **Exponential fit R² = 0.7316** (slightly better)
- **Power law fit R² = 0.7269**
- Half-life of ~183 days (time halves every 6 months)
- Total improvement: 45 min → 1.92 min (23.4x speedup over ~580 days)

### Anomalies
- **Record #22** shows slight time increase from #21 (2.933 → 2.990 min)
  - Explained by timing rule change after record #21
  - Removed "grace period" and banned certain compiler flags

### PR Date vs Listed Date
- Listed dates in README closely match **PR creation dates** (mean diff: 0.5 days)
- Large discrepancy with merge dates (mean diff: -17.5 days)
- This means records are "announced" when PR is opened, not when merged

### Code Growth
- Started at ~446 LoC (record #2)
- Grew to ~1779 LoC (record #55)
- 4x increase in code size for 23x improvement in speed

## Limitations

1. **Record #1 has no LoC data** - it's the llm.c baseline which is in a different repo

2. **Star history requires manual export** - star-history.com doesn't have an easy API

3. **Early records (1-21) don't have linked PRs** - improvements were made via direct commits

4. **LoC is a rough proxy for effort** - doesn't account for:
   - Deleted code (refactoring)
   - Experimental work that was later reverted
   - Time spent on research/experimentation

5. **Date mapping uses git commit dates** - may not perfectly align with when code was actually written

6. **Only counts train_gpt.py** - some improvements may involve other files like:
   - `muon_optimizer.py` (but this is usually imported)
   - Triton kernels
   - Data loading scripts

## Future Work (Not Implemented)

- [ ] Normalize times for hardware differences
- [ ] Track lines of code for individual changes (deltas)
- [ ] Regress on individual change data (similar to Tamay's analysis)
- [ ] Incorporate GPT-2 Medium track (`train_gpt_medium.py`)
- [ ] More statistical modeling (confidence intervals, etc.)
- [ ] Automated tests

## File Structure

```
nanogpt-analysis/
├── src/
│   ├── extract_records.py   # Parse README table
│   ├── loc_analysis.py      # Count lines of code
│   ├── analysis.py          # Compile all data
│   ├── plots.py             # Create visualizations
│   └── pr_date_analysis.py  # Compare PR vs listed dates
├── data/
│   ├── modded-nanogpt/      # Cloned repo
│   ├── raw_records.json     # Extracted timing records
│   ├── loc_data.json        # Lines of code per record
│   ├── improvements.json    # Compiled analysis data
│   ├── metrics.yaml         # Regression results
│   └── pr_date_analysis.json
├── plots/
│   ├── speed_over_time.png
│   ├── speed_vs_loc.png
│   └── loc_over_time.png
└── ai_notes.md              # This file
```

## Running the Analysis

```bash
# Install dependencies
uv add pandas matplotlib numpy scipy pyyaml

# Run in order:
uv run python src/extract_records.py
uv run python src/loc_analysis.py
uv run python src/analysis.py
uv run python src/plots.py

# Optional: PR date analysis (requires gh CLI)
uv run python src/pr_date_analysis.py
```

## References

- [NanoGPT Speedrun Repository](https://github.com/KellerJordan/modded-nanogpt)
- [Tamay's analysis thread](https://x.com/tamaybes/status/1890263324899848412)
- [LessWrong writeup](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)
