#!/usr/bin/env python3
"""
Comparar resultados de experimentos de mejora vs baseline.
Uso: python3 compare_results.py
"""

import json
import glob
import os

def load_metrics(filepath):
    """Load metrics from JSON file."""
    with open(filepath) as f:
        return json.load(f)

def main():
    print("=" * 70)
    print("  COMPARACIÓN: Experimentos de Mejora vs Baseline")
    print("=" * 70)

    # Find baseline results
    baseline_dir = "../results/safe_20251129_215918"
    baseline_file = f"{baseline_dir}/cfg01_phaseA_default_s42_metrics.json"

    if os.path.exists(baseline_file):
        baseline = load_metrics(baseline_file)
        b_f1 = baseline['baseline']['macro_f1']
        b_aug = baseline['augmented']['macro_f1']
        b_synth = baseline.get('synthetic_data', {}).get('accepted_count', 0)
        b_delta = (b_aug - b_f1) / b_f1 * 100

        print(f"\n📊 BASELINE (cfg01_phaseA_default_s42):")
        print(f"   Baseline F1:  {b_f1:.4f}")
        print(f"   Augmented F1: {b_aug:.4f}")
        print(f"   Delta:        {b_delta:+.2f}%")
        print(f"   Sintéticos:   {b_synth}")
    else:
        print(f"\n⚠️  No se encontró baseline: {baseline_file}")
        b_delta = 0

    # Find improvement experiment results
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"\n⚠️  No hay resultados de mejoras aún. Ejecuta:")
        print("   ./run_both_experiments.sh")
        return

    improvements = glob.glob(f"{results_dir}/*_metrics.json")

    if not improvements:
        print(f"\n⚠️  No hay resultados de mejoras aún. Ejecuta:")
        print("   ./run_both_experiments.sh")
        return

    print("\n" + "-" * 70)
    print("  EXPERIMENTOS DE MEJORA")
    print("-" * 70)

    for mf in sorted(improvements):
        exp_name = os.path.basename(mf).replace("_metrics.json", "")
        d = load_metrics(mf)

        exp_b = d['baseline']['macro_f1']
        exp_a = d['augmented']['macro_f1']
        exp_s = d.get('synthetic_data', {}).get('accepted_count', 0)
        exp_delta = (exp_a - exp_b) / exp_b * 100

        improvement_vs_baseline = exp_delta - b_delta

        print(f"\n📊 {exp_name}:")
        print(f"   Baseline F1:  {exp_b:.4f}")
        print(f"   Augmented F1: {exp_a:.4f}")
        print(f"   Delta:        {exp_delta:+.2f}%")
        print(f"   Sintéticos:   {exp_s}")
        print(f"   vs Baseline:  {improvement_vs_baseline:+.2f}% {'✅' if improvement_vs_baseline > 0 else '❌'}")

    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)

    # Summary table
    print(f"\n{'Experimento':<30} {'Delta%':>10} {'Sintéticos':>12} {'vs Base':>10}")
    print("-" * 70)
    print(f"{'Baseline (cfg01))':<30} {b_delta:>+9.2f}% {b_synth:>12} {'---':>10}")

    for mf in sorted(improvements):
        exp_name = os.path.basename(mf).replace("_metrics.json", "").split("_2024")[0]
        d = load_metrics(mf)
        exp_b = d['baseline']['macro_f1']
        exp_a = d['augmented']['macro_f1']
        exp_s = d.get('synthetic_data', {}).get('accepted_count', 0)
        exp_delta = (exp_a - exp_b) / exp_b * 100
        vs_base = exp_delta - b_delta

        marker = "✅" if vs_base > 0.5 else "➡️" if vs_base > -0.5 else "❌"
        print(f"{exp_name:<30} {exp_delta:>+9.2f}% {exp_s:>12} {vs_base:>+9.2f}% {marker}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
