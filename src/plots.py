"""Create plots and fit regression models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats
from scipy.optimize import curve_fit


def load_improvements(path: str) -> list[dict]:
    """Load improvements data."""
    with open(path, "r") as f:
        return json.load(f)


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def exponential(x, a, b):
    """Exponential decay: y = a * exp(b * x)"""
    return a * np.exp(b * x)


def fit_models(days: np.ndarray, times: np.ndarray) -> dict:
    """Fit power law and exponential models to the data."""
    results = {}

    # For fitting, we need positive days (shift to avoid x=0)
    days_shifted = days + 1  # Shift by 1 to avoid log(0)

    # Power law fit (in log-log space for better fitting)
    log_days = np.log(days_shifted)
    log_times = np.log(times)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_days, log_times)
    results["power_law"] = {
        "exponent": float(slope),
        "coefficient": float(np.exp(intercept)),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "formula": f"time = {np.exp(intercept):.4f} * (days+1)^{slope:.4f}",
    }

    # Exponential fit (in semi-log space)
    slope_exp, intercept_exp, r_value_exp, p_value_exp, std_err_exp = stats.linregress(
        days, log_times
    )
    results["exponential"] = {
        "decay_rate": float(slope_exp),
        "initial_value": float(np.exp(intercept_exp)),
        "r_squared": float(r_value_exp**2),
        "p_value": float(p_value_exp),
        "formula": f"time = {np.exp(intercept_exp):.4f} * exp({slope_exp:.6f} * days)",
    }

    # Calculate half-life for exponential decay
    if slope_exp < 0:
        half_life = -np.log(2) / slope_exp
        results["exponential"]["half_life_days"] = float(half_life)

    return results


def plot_speed_over_time(improvements: list[dict], output_dir: Path, fit_results: dict):
    """Plot log(speed) over time with fitted curves."""
    # Filter records with valid data
    valid = [i for i in improvements if i["days_since_start"] is not None]

    days = np.array([i["days_since_start"] for i in valid])
    times = np.array([i["record_time_minutes"] for i in valid])
    record_nums = [i["record_num"] for i in valid]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time vs Days (linear scale)
    ax1 = axes[0, 0]
    ax1.scatter(days, times, alpha=0.7, s=30)
    ax1.set_xlabel("Days since first record")
    ax1.set_ylabel("Training time (minutes)")
    ax1.set_title("Training Time vs Calendar Time")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log(time) vs Days (semi-log - for exponential fit)
    ax2 = axes[0, 1]
    ax2.scatter(days, times, alpha=0.7, s=30)
    ax2.set_yscale("log")
    ax2.set_xlabel("Days since first record")
    ax2.set_ylabel("Training time (minutes, log scale)")
    ax2.set_title("Training Time vs Days (Semi-log)")
    ax2.grid(True, alpha=0.3)

    # Add exponential fit line
    days_fit = np.linspace(0, max(days), 100)
    exp_fit = fit_results["exponential"]["initial_value"] * np.exp(
        fit_results["exponential"]["decay_rate"] * days_fit
    )
    ax2.plot(days_fit, exp_fit, "r-", alpha=0.7, label=f"Exponential fit (R²={fit_results['exponential']['r_squared']:.3f})")
    ax2.legend()

    # Plot 3: Log(time) vs Log(days) (log-log - for power law fit)
    ax3 = axes[1, 0]
    # Filter out day 0 for log-log
    mask = days > 0
    ax3.scatter(days[mask], times[mask], alpha=0.7, s=30)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Days since first record (log scale)")
    ax3.set_ylabel("Training time (minutes, log scale)")
    ax3.set_title("Training Time vs Days (Log-log)")
    ax3.grid(True, alpha=0.3)

    # Add power law fit line
    days_fit_log = np.linspace(1, max(days), 100)
    power_fit = fit_results["power_law"]["coefficient"] * np.power(
        days_fit_log + 1, fit_results["power_law"]["exponent"]
    )
    ax3.plot(days_fit_log, power_fit, "g-", alpha=0.7, label=f"Power law fit (R²={fit_results['power_law']['r_squared']:.3f})")
    ax3.legend()

    # Plot 4: Record number progression
    ax4 = axes[1, 1]
    ax4.scatter(record_nums, times, alpha=0.7, s=30)
    ax4.set_xlabel("Record number")
    ax4.set_ylabel("Training time (minutes)")
    ax4.set_title("Training Time vs Record Number")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "speed_over_time.png", dpi=150)
    plt.close()

    print(f"Saved speed_over_time.png")


def plot_speed_vs_loc(improvements: list[dict], output_dir: Path):
    """Plot speed vs lines of code."""
    # Filter records with valid data
    valid = [i for i in improvements if i["cumulative_loc"] is not None and i["record_time_minutes"] is not None]

    loc = np.array([i["cumulative_loc"] for i in valid])
    times = np.array([i["record_time_minutes"] for i in valid])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1 = axes[0]
    ax1.scatter(loc, times, alpha=0.7, s=30)
    ax1.set_xlabel("Cumulative Lines of Code")
    ax1.set_ylabel("Training time (minutes)")
    ax1.set_title("Training Time vs Lines of Code")
    ax1.grid(True, alpha=0.3)

    # Semi-log scale
    ax2 = axes[1]
    ax2.scatter(loc, times, alpha=0.7, s=30)
    ax2.set_yscale("log")
    ax2.set_xlabel("Cumulative Lines of Code")
    ax2.set_ylabel("Training time (minutes, log scale)")
    ax2.set_title("Training Time vs LoC (Semi-log)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_loc.png", dpi=150)
    plt.close()

    print(f"Saved speed_vs_loc.png")


def plot_speed_vs_stars(improvements: list[dict], output_dir: Path):
    """Plot speed vs stars (if available)."""
    # Filter records with valid data
    valid = [i for i in improvements if i["stars"] is not None and i["record_time_minutes"] is not None]

    if not valid:
        print("No star data available, skipping speed_vs_stars plot")
        return

    stars = np.array([i["stars"] for i in valid])
    times = np.array([i["record_time_minutes"] for i in valid])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1 = axes[0]
    ax1.scatter(stars, times, alpha=0.7, s=30)
    ax1.set_xlabel("Repository Stars")
    ax1.set_ylabel("Training time (minutes)")
    ax1.set_title("Training Time vs Stars")
    ax1.grid(True, alpha=0.3)

    # Semi-log on Y
    ax2 = axes[1]
    ax2.scatter(stars, times, alpha=0.7, s=30)
    ax2.set_yscale("log")
    ax2.set_xlabel("Repository Stars")
    ax2.set_ylabel("Training time (minutes, log scale)")
    ax2.set_title("Training Time vs Stars (Semi-log)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_stars.png", dpi=150)
    plt.close()

    print(f"Saved speed_vs_stars.png")


def plot_loc_over_time(improvements: list[dict], output_dir: Path):
    """Plot lines of code over time."""
    valid = [i for i in improvements if i["cumulative_loc"] is not None and i["days_since_start"] is not None]

    days = np.array([i["days_since_start"] for i in valid])
    loc = np.array([i["cumulative_loc"] for i in valid])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(days, loc, alpha=0.7, s=30)
    ax.set_xlabel("Days since first record")
    ax.set_ylabel("Cumulative Lines of Code (train_gpt.py)")
    ax.set_title("Code Growth Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "loc_over_time.png", dpi=150)
    plt.close()

    print(f"Saved loc_over_time.png")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    plots_dir = Path(__file__).parent.parent / "plots"
    improvements_path = data_dir / "improvements.json"
    metrics_path = data_dir / "metrics.yaml"

    plots_dir.mkdir(exist_ok=True)

    improvements = load_improvements(str(improvements_path))
    print(f"Loaded {len(improvements)} improvements")

    # Filter for fitting (need days_since_start)
    valid_for_fit = [i for i in improvements if i["days_since_start"] is not None]
    days = np.array([i["days_since_start"] for i in valid_for_fit])
    times = np.array([i["record_time_minutes"] for i in valid_for_fit])

    # Fit models
    print("\nFitting models...")
    fit_results = fit_models(days, times)

    print(f"\nPower Law: {fit_results['power_law']['formula']}")
    print(f"  R² = {fit_results['power_law']['r_squared']:.4f}")

    print(f"\nExponential: {fit_results['exponential']['formula']}")
    print(f"  R² = {fit_results['exponential']['r_squared']:.4f}")
    if "half_life_days" in fit_results["exponential"]:
        print(f"  Half-life = {fit_results['exponential']['half_life_days']:.1f} days")

    # Save metrics
    metrics = {
        "data_summary": {
            "total_records": len(improvements),
            "records_with_loc": sum(1 for i in improvements if i["cumulative_loc"] is not None),
            "records_with_stars": sum(1 for i in improvements if i["stars"] is not None),
            "date_range": {
                "start": improvements[0]["date"],
                "end": improvements[-1]["date"],
                "total_days": int(days[-1]),
            },
            "time_range": {
                "initial_minutes": float(times[0]),
                "final_minutes": float(times[-1]),
                "improvement_factor": float(times[0] / times[-1]),
            },
        },
        "regression_results": fit_results,
    }

    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved metrics to {metrics_path}")

    # Create plots
    print("\nGenerating plots...")
    plot_speed_over_time(improvements, plots_dir, fit_results)
    plot_speed_vs_loc(improvements, plots_dir)
    plot_speed_vs_stars(improvements, plots_dir)
    plot_loc_over_time(improvements, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")

    # Visual inspection notes
    print("\n" + "=" * 60)
    print("VISUAL INSPECTION NOTES:")
    print("=" * 60)

    # Check for anomalies
    anomalies = []

    # Check for large jumps
    for i in range(1, len(valid_for_fit)):
        prev_time = times[i - 1]
        curr_time = times[i]
        if prev_time / curr_time > 2:  # More than 2x improvement
            anomalies.append(
                f"Record #{valid_for_fit[i]['record_num']}: Large jump from {prev_time:.2f} to {curr_time:.2f} min "
                f"({prev_time/curr_time:.1f}x improvement)"
            )

    # Check for regressions (time increased)
    for i in range(1, len(valid_for_fit)):
        if times[i] > times[i - 1]:
            anomalies.append(
                f"Record #{valid_for_fit[i]['record_num']}: Time increased from {times[i-1]:.3f} to {times[i]:.3f} min"
            )

    if anomalies:
        print("\nAnomalies detected:")
        for a in anomalies:
            print(f"  - {a}")
    else:
        print("\nNo major anomalies detected in the progression.")

    # Model comparison
    print("\nModel Comparison:")
    if fit_results["exponential"]["r_squared"] > fit_results["power_law"]["r_squared"]:
        print(f"  Exponential fit is better (R²={fit_results['exponential']['r_squared']:.4f} vs {fit_results['power_law']['r_squared']:.4f})")
    else:
        print(f"  Power law fit is better (R²={fit_results['power_law']['r_squared']:.4f} vs {fit_results['exponential']['r_squared']:.4f})")


if __name__ == "__main__":
    main()
