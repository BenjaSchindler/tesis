#!/usr/bin/env python3
"""
Calculate component contributions using existing K-fold results.

We have:
- ENS_Top3_G5: CMB3 + CF1 + V4 + G5 = +1.29%
- ENS_Top3: CMB3 + CF1 + V4 = +1.22%
- ENS_CMB3_CF1: CMB3 + CF1 = +0.87%
- ENS_CMB3_V2: CMB3 + V2 = +0.93%
- ENS_CMB3_G5: CMB3 + G5 = +0.80%
- CMB3_skip alone = +0.57%
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results"

def load_kfold_delta(config):
    """Load delta from kfold results."""
    # Try main results dir
    path = RESULTS_DIR / f"{config}_s42_kfold_k5.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data["delta"]["mean"] * 100

    # Try variance tests
    path = RESULTS_DIR / "Variance_tests" / f"{config}_kfold.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data["delta"]["mean"] * 100

    return None

def main():
    print("="*70)
    print("CONTRIBUTION ANALYSIS FROM EXISTING K-FOLD RESULTS")
    print("="*70)

    # Load existing results
    results = {}
    configs = [
        "ENS_Top3_G5",
        "ENS_Top3",
        "ENS_CMB3_CF1",
        "ENS_CMB3_V2",
        "ENS_CMB3_G5",
        "CMB3_skip",
        "CF1_conf_band",
        "V4_ultra",
        "G5_K25_medium",
        "V2_high_vol"
    ]

    for config in configs:
        delta = load_kfold_delta(config)
        if delta is not None:
            results[config] = delta
            print(f"  {config}: {delta:+.3f}%")

    print("\n" + "="*70)
    print("CALCULATED CONTRIBUTIONS")
    print("="*70)

    # G5 contribution (ENS_Top3_G5 - ENS_Top3)
    if "ENS_Top3_G5" in results and "ENS_Top3" in results:
        g5_contrib = results["ENS_Top3_G5"] - results["ENS_Top3"]
        print(f"\nG5_K25_medium contribution:")
        print(f"  ENS_Top3_G5 ({results['ENS_Top3_G5']:+.3f}%) - ENS_Top3 ({results['ENS_Top3']:+.3f}%)")
        print(f"  = {g5_contrib:+.3f}%")

    # V4 contribution (ENS_Top3 - ENS_CMB3_CF1)
    if "ENS_Top3" in results and "ENS_CMB3_CF1" in results:
        v4_contrib = results["ENS_Top3"] - results["ENS_CMB3_CF1"]
        print(f"\nV4_ultra contribution:")
        print(f"  ENS_Top3 ({results['ENS_Top3']:+.3f}%) - ENS_CMB3_CF1 ({results['ENS_CMB3_CF1']:+.3f}%)")
        print(f"  = {v4_contrib:+.3f}%")

    # CF1 contribution to CMB3 (ENS_CMB3_CF1 - CMB3_skip)
    if "ENS_CMB3_CF1" in results and "CMB3_skip" in results:
        cf1_contrib = results["ENS_CMB3_CF1"] - results["CMB3_skip"]
        print(f"\nCF1_conf_band contribution (to CMB3):")
        print(f"  ENS_CMB3_CF1 ({results['ENS_CMB3_CF1']:+.3f}%) - CMB3_skip ({results['CMB3_skip']:+.3f}%)")
        print(f"  = {cf1_contrib:+.3f}%")

    # G5 contribution to CMB3 (ENS_CMB3_G5 - CMB3_skip)
    if "ENS_CMB3_G5" in results and "CMB3_skip" in results:
        g5_to_cmb3 = results["ENS_CMB3_G5"] - results["CMB3_skip"]
        print(f"\nG5_K25_medium contribution (to CMB3):")
        print(f"  ENS_CMB3_G5 ({results['ENS_CMB3_G5']:+.3f}%) - CMB3_skip ({results['CMB3_skip']:+.3f}%)")
        print(f"  = {g5_to_cmb3:+.3f}%")

    # V2 contribution to CMB3 (ENS_CMB3_V2 - CMB3_skip)
    if "ENS_CMB3_V2" in results and "CMB3_skip" in results:
        v2_to_cmb3 = results["ENS_CMB3_V2"] - results["CMB3_skip"]
        print(f"\nV2_high_vol contribution (to CMB3):")
        print(f"  ENS_CMB3_V2 ({results['ENS_CMB3_V2']:+.3f}%) - CMB3_skip ({results['CMB3_skip']:+.3f}%)")
        print(f"  = {v2_to_cmb3:+.3f}%")

    print("\n" + "="*70)
    print("RANKING BY CONTRIBUTION TO FULL ENSEMBLE")
    print("="*70)

    contributions = []

    # Calculate all contributions
    if "ENS_Top3_G5" in results and "ENS_Top3" in results:
        contributions.append(("G5_K25_medium", results["ENS_Top3_G5"] - results["ENS_Top3"]))

    if "ENS_Top3" in results and "ENS_CMB3_CF1" in results:
        contributions.append(("V4_ultra", results["ENS_Top3"] - results["ENS_CMB3_CF1"]))

    if "ENS_CMB3_CF1" in results and "CMB3_skip" in results:
        contributions.append(("CF1_conf_band", results["ENS_CMB3_CF1"] - results["CMB3_skip"]))

    # CMB3 baseline
    if "CMB3_skip" in results:
        contributions.append(("CMB3_skip (base)", results["CMB3_skip"]))

    # Sort by contribution
    contributions.sort(key=lambda x: -x[1])

    print(f"\n{'Rank':<6} {'Component':<20} {'Contribution':>12}")
    print("-"*40)
    for i, (comp, contrib) in enumerate(contributions, 1):
        print(f"{i:<6} {comp:<20} {contrib:>+11.3f}%")

    # Analysis
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    if contributions:
        most = contributions[0]
        least = contributions[-1]

        print(f"""
    1. CMB3_skip provides the BASE contribution of +0.57%

    2. Adding CF1_conf_band to CMB3 adds +{results.get('ENS_CMB3_CF1', 0) - results.get('CMB3_skip', 0):.2f}%

    3. Adding V4_ultra to CMB3+CF1 adds +{results.get('ENS_Top3', 0) - results.get('ENS_CMB3_CF1', 0):.2f}%
       (V4 is the MOST VALUABLE addition!)

    4. Adding G5_K25_medium to Top3 adds only +{results.get('ENS_Top3_G5', 0) - results.get('ENS_Top3', 0):.2f}%
       (G5 provides MINIMAL additional value)

    RECOMMENDATION:
    - ENS_Top3 (CMB3 + CF1 + V4) is nearly as good as ENS_Top3_G5
    - Consider dropping G5 to save 53 synthetics for only 0.07% loss
    - V4 provides the best marginal improvement
    """)

    # Save results
    output = {
        "raw_results": results,
        "contributions": {c[0]: c[1] for c in contributions},
        "recommendation": "ENS_Top3 (CMB3 + CF1 + V4) is optimal, G5 adds minimal value"
    }

    with open(OUTPUT_DIR / "contribution_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {OUTPUT_DIR / 'contribution_analysis.json'}")

if __name__ == "__main__":
    main()
