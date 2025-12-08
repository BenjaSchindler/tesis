#!/usr/bin/env python3
"""
Phase F Analysis: Diminishing Returns - Marginal improvement per synthetic
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results"

def load_kfold_data():
    """Load kfold results sorted by synthetics."""
    results = []

    # Main results
    for f in RESULTS_DIR.glob("*_kfold_k5.json"):
        if f.stat().st_size == 0:
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append({
                "config": data.get("config", f.stem),
                "n_synthetic": data.get("n_synthetic", 0),
                "delta_pct": data["delta"]["mean"] * 100,
                "type": "ensemble" if "ENS_" in f.stem else "single"
            })
        except:
            continue

    # Variance tests
    vt_dir = RESULTS_DIR / "Variance_tests"
    if vt_dir.exists():
        for f in vt_dir.glob("*_kfold.json"):
            if f.stat().st_size == 0:
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
                config = data.get("config", f.stem.replace("_kfold", ""))
                if any(r["config"] == config for r in results):
                    continue
                results.append({
                    "config": config,
                    "n_synthetic": data.get("n_synthetic", 0),
                    "delta_pct": data["delta"]["mean"] * 100,
                    "type": "single"
                })
            except:
                continue

    return sorted(results, key=lambda x: x["n_synthetic"])

def analyze_diminishing_returns():
    """Calculate marginal returns for each step."""
    data = load_kfold_data()

    # Filter out zero synthetics
    data = [d for d in data if d["n_synthetic"] > 0]

    print("\n" + "="*90)
    print("DIMINISHING RETURNS ANALYSIS")
    print("="*90)

    print("\n### Ordered by Synthetics Count")
    print("-"*90)
    print(f"{'From':<20} {'To':<20} {'ΔSynth':>8} {'From Δ%':>10} {'To Δ%':>10} {'Marginal':>12}")
    print("-"*90)

    # Track marginal returns
    marginals = []
    ensemble_transitions = []

    for i in range(len(data) - 1):
        curr = data[i]
        next_ = data[i + 1]

        d_synth = next_["n_synthetic"] - curr["n_synthetic"]
        d_delta = next_["delta_pct"] - curr["delta_pct"]

        if d_synth > 0:
            marginal = d_delta / d_synth
            marginals.append({
                "from": curr["config"],
                "to": next_["config"],
                "d_synth": d_synth,
                "d_delta": d_delta,
                "marginal": marginal,
                "from_type": curr["type"],
                "to_type": next_["type"]
            })

            # Highlight ensemble transitions
            marker = ""
            if curr["type"] == "single" and next_["type"] == "ensemble":
                marker = " ← ENSEMBLE JUMP"
                ensemble_transitions.append(marginals[-1])

            color = "+" if marginal > 0 else ""
            print(f"{curr['config'][:19]:<20} {next_['config'][:19]:<20} {d_synth:>+8} "
                  f"{curr['delta_pct']:>+9.2f}% {next_['delta_pct']:>+9.2f}% {marginal:>+11.4f}%/syn{marker}")

    # Summary statistics
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90)

    positive_marginals = [m for m in marginals if m["marginal"] > 0]
    negative_marginals = [m for m in marginals if m["marginal"] < 0]

    print(f"\nPositive transitions: {len(positive_marginals)}/{len(marginals)} ({100*len(positive_marginals)/len(marginals):.0f}%)")
    print(f"Negative transitions: {len(negative_marginals)}/{len(marginals)} ({100*len(negative_marginals)/len(marginals):.0f}%)")

    if positive_marginals:
        avg_positive = sum(m["marginal"] for m in positive_marginals) / len(positive_marginals)
        best = max(positive_marginals, key=lambda x: x["marginal"])
        print(f"\nAverage positive marginal: {avg_positive:+.4f}%/synthetic")
        print(f"Best transition: {best['from']} → {best['to']}: {best['marginal']:+.4f}%/syn")

    if negative_marginals:
        avg_negative = sum(m["marginal"] for m in negative_marginals) / len(negative_marginals)
        worst = min(negative_marginals, key=lambda x: x["marginal"])
        print(f"\nAverage negative marginal: {avg_negative:+.4f}%/synthetic")
        print(f"Worst transition: {worst['from']} → {worst['to']}: {worst['marginal']:+.4f}%/syn")

    # Ensemble analysis
    print("\n" + "="*90)
    print("ENSEMBLE JUMP ANALYSIS")
    print("="*90)

    if ensemble_transitions:
        for et in ensemble_transitions:
            print(f"\n{et['from']} (single) → {et['to']} (ensemble)")
            print(f"  Synthetics: +{et['d_synth']}")
            print(f"  Delta improvement: {et['d_delta']:+.3f}%")
            print(f"  Marginal: {et['marginal']:+.4f}%/synthetic")
    else:
        print("\nNo direct single→ensemble transitions found")

    # Key insight: Last ensemble step
    print("\n" + "="*90)
    print("KEY INSIGHT: Last Ensemble Step")
    print("="*90)

    ensemble_data = [d for d in data if d["type"] == "ensemble"]
    if len(ensemble_data) >= 2:
        second_last = ensemble_data[-2]
        last = ensemble_data[-1]

        d_synth = last["n_synthetic"] - second_last["n_synthetic"]
        d_delta = last["delta_pct"] - second_last["delta_pct"]
        marginal = d_delta / d_synth if d_synth > 0 else 0

        print(f"\n{second_last['config']} → {last['config']}")
        print(f"  Added synthetics: +{d_synth}")
        print(f"  Added improvement: {d_delta:+.3f}%")
        print(f"  Marginal return: {marginal:+.5f}%/synthetic")

        if marginal < 0.005:
            print(f"\n  ⚠ DIMINISHING RETURNS DETECTED!")
            print(f"  Adding {d_synth} synthetics only gained {d_delta:+.3f}% improvement")
            print(f"  This is {marginal*1000:.2f}‰ (per-mille) per synthetic")

    # Save results
    output = {
        "marginals": marginals,
        "positive_count": len(positive_marginals),
        "negative_count": len(negative_marginals),
        "ensemble_transitions": ensemble_transitions
    }

    with open(OUTPUT_DIR / "diminishing_returns.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {OUTPUT_DIR / 'diminishing_returns.json'}")

if __name__ == "__main__":
    analyze_diminishing_returns()
