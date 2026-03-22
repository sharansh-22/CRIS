"""
entropy_comparison.py — Dynamic WandB Experiment Runner for TS-001

Calls run_entropy_method_selection() from entropy.py, extracts all metrics
from the results dict, and logs them live to Weights & Biases with a rich
dashboard: summary metrics, per-method tables, ranking chart, per-event
breakdowns, and system resource tracking.

NO HARDCODED NUMBERS — every value comes from the results variable.
"""

import os
import wandb
import psutil
from pathlib import Path

from layer3_bssc.engine.entropy import run_entropy_method_selection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "simulation_output"


def main():
    # ==================================================================
    # 1. Initialise WandB Run
    # ==================================================================
    csv_path = str(DATA_DIR / "Indices" / "SPY.csv")

    run = wandb.init(
        project="CRIS",
        job_type="validation",
        name="TS-001-Entropy-Comparison",
        tags=["ts-001", "entropy", "method-selection", "SPY"],
        notes="Empirical comparison of 4 entropy methods across 3 historical crises. "
              "Selects primary + confirmation method for black swan detection.",
        config={
            "ticker": "SPY",
            "csv_path": csv_path,
            "test_id": "TS-001",
            "events": ["covid_2020", "q4_2018_selloff", "fed_2022"],
            "methods": ["shannon", "permutation", "sample", "tsallis"],
            "scoring_weights": {
                "lead_time": 0.40,
                "false_positive_rate": 0.30,
                "magnitude": 0.20,
                "consistency": 0.10,
            },
        },
    )

    # ==================================================================
    # 2. Run the Entropy Method Selection Engine
    # ==================================================================
    print("=" * 60)
    print("  CRIS TS-001 — Entropy Method Comparison")
    print("  Running run_entropy_method_selection('SPY', ...)")
    print("=" * 60)

    results = run_entropy_method_selection("SPY", csv_path)
    metrics_table = results["metrics_table"]
    methods = list(metrics_table.keys())

    print(f"\nEngine completed. Primary: {results['primary_method']}, "
          f"Confirmation: {results['confirmation_method']}")

    # ==================================================================
    # 3. Log Summary Metrics — Top-Level Dashboard Cards
    # ==================================================================
    summary = {
        # Winner info
        "winner/method": results["primary_method"],
        "winner/composite_score": results["primary_score"],
        "winner/lead_time_days": metrics_table[results["primary_method"]]["lead_time"],

        # Confirmation info
        "confirmation/method": results["confirmation_method"],
        "confirmation/composite_score": results["confirmation_score"],

        # Overview counts
        "overview/methods_evaluated": len(methods),
        "overview/events_tested": len(results["evaluated_on"]),
        "overview/methods_rejected": len(results["rejected_methods"]),
    }
    wandb.log(summary)

    # ==================================================================
    # 4. Log Per-Method Metrics — Individual Metric Cards
    # ==================================================================
    for method, m in metrics_table.items():
        wandb.log({
            f"scores/{method}_composite": m["score"],
            f"lead_time/{method}_days": m["lead_time"],
            f"false_positives/{method}_per_month": m["false_pos_rate"],
            f"magnitude/{method}_sigma": m["magnitude"],
            f"consistency/{method}": m["consistency"],
        })

    # ==================================================================
    # 5. Main Comparison Table — All Methods Side-by-Side
    # ==================================================================
    comparison_table = wandb.Table(
        columns=[
            "Method",
            "Lead Time (days)",
            "FP Rate (/mo)",
            "Magnitude (σ)",
            "Consistency",
            "Composite Score",
            "Role",
        ],
        data=[
            [
                method.capitalize(),
                round(m["lead_time"], 1),
                round(m["false_pos_rate"], 3),
                round(m["magnitude"], 3),
                round(m["consistency"], 2),
                round(m["score"], 3),
                "🥇 Primary" if method == results["primary_method"]
                else "🥈 Confirmation" if method == results["confirmation_method"]
                else "❌ Rejected",
            ]
            for method, m in metrics_table.items()
        ],
    )
    wandb.log({"comparison/full_metrics_table": comparison_table})

    # ==================================================================
    # 6. Ranking Table — Sorted by Composite Score
    # ==================================================================
    sorted_methods = sorted(
        metrics_table.items(), key=lambda x: x[1]["score"], reverse=True
    )

    ranking_table = wandb.Table(
        columns=["Rank", "Method", "Composite Score", "Status"],
        data=[
            [
                i + 1,
                method.capitalize(),
                round(m["score"], 3),
                "✅ Selected" if method in (results["primary_method"], results["confirmation_method"])
                else "❌ Rejected",
            ]
            for i, (method, m) in enumerate(sorted_methods)
        ],
    )
    wandb.log({"comparison/ranking_table": ranking_table})

    # ==================================================================
    # 7. Rejection Table — Why Methods Were Rejected
    # ==================================================================
    if results["rejected_methods"]:
        rejection_table = wandb.Table(
            columns=["Method", "Composite Score", "Reason"],
            data=[
                [
                    method.capitalize(),
                    round(metrics_table[method]["score"], 3),
                    reason,
                ]
                for method, reason in results["rejected_methods"].items()
            ],
        )
        wandb.log({"comparison/rejection_reasons": rejection_table})

    # ==================================================================
    # 8. Bar Charts — Visual Comparison of Each Metric
    # ==================================================================
    # Composite Score Bar Chart
    score_chart_data = [
        [method.capitalize(), round(m["score"], 3)]
        for method, m in sorted_methods
    ]
    score_table = wandb.Table(columns=["Method", "Composite Score"], data=score_chart_data)
    wandb.log({
        "charts/composite_scores": wandb.plot.bar(
            score_table, "Method", "Composite Score",
            title="Composite Score by Method"
        )
    })

    # Lead Time Bar Chart
    lt_data = [
        [method.capitalize(), round(m["lead_time"], 1)]
        for method, m in metrics_table.items()
    ]
    lt_table = wandb.Table(columns=["Method", "Lead Time (days)"], data=lt_data)
    wandb.log({
        "charts/lead_times": wandb.plot.bar(
            lt_table, "Method", "Lead Time (days)",
            title="Crisis Lead Time by Method (days before trough)"
        )
    })

    # False Positive Rate Bar Chart
    fp_data = [
        [method.capitalize(), round(m["false_pos_rate"], 3)]
        for method, m in metrics_table.items()
    ]
    fp_table = wandb.Table(columns=["Method", "FP Rate (/mo)"], data=fp_data)
    wandb.log({
        "charts/false_positive_rates": wandb.plot.bar(
            fp_table, "Method", "FP Rate (/mo)",
            title="False Positive Rate by Method (lower is better)"
        )
    })

    # ==================================================================
    # 9. Decision Summary Table
    # ==================================================================
    decision_table = wandb.Table(
        columns=["Field", "Value"],
        data=[
            ["Primary Method", results["primary_method"].capitalize()],
            ["Primary Score", str(round(results["primary_score"], 3))],
            ["Confirmation Method", results["confirmation_method"].capitalize()],
            ["Confirmation Score", str(round(results["confirmation_score"], 3))],
            ["Selection Rationale", results["selection_rationale"]],
            ["Events Evaluated", ", ".join(results["evaluated_on"])],
        ],
    )
    wandb.log({"decision/summary": decision_table})

    # ==================================================================
    # 10. System Resource Tracking
    # ==================================================================
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())

    system_metrics = {
        "system/ram_total_gb": round(mem.total / (1024 ** 3), 2),
        "system/ram_used_gb": round(mem.used / (1024 ** 3), 2),
        "system/ram_available_gb": round(mem.available / (1024 ** 3), 2),
        "system/ram_percent": mem.percent,
        "system/process_rss_mb": round(proc.memory_info().rss / (1024 ** 2), 2),
    }
    wandb.log(system_metrics)

    print(
        f"\n[System] RAM: {mem.total / (1024**3):.1f} GB total, "
        f"{mem.percent}% used | Process RSS: "
        f"{proc.memory_info().rss / (1024**2):.1f} MB"
    )

    # ==================================================================
    # 11. Upload Remaining Artifacts
    # ==================================================================
    # Upload the JSON selection output as an artifact
    json_path = OUTPUT_DIR / "entropy_method_selection.json"
    if json_path.exists():
        artifact = wandb.Artifact("entropy-selection-results", type="results")
        artifact.add_file(str(json_path))
        wandb.log_artifact(artifact)
        print(f"[Artifact] Uploaded {json_path.name}")

    # Upload any remaining plots
    plot_path = OUTPUT_DIR / "jump_diffusion_SPY.png"
    if plot_path.exists():
        wandb.log({"plots/jump_diffusion": wandb.Image(str(plot_path))})
        print(f"[Plot] Uploaded {plot_path.name}")

    # ==================================================================
    # 12. Finish
    # ==================================================================
    wandb.finish()
    print("\n" + "=" * 60)
    print("  TS-001 Complete — Results logged to WandB")
    print("=" * 60)


if __name__ == "__main__":
    main()
