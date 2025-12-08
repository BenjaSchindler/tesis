#!/usr/bin/env python3
"""
Crear ensembles desde replicación.
Combina sintéticos de múltiples configs en los ensembles finales.
"""
import pandas as pd
import argparse
from pathlib import Path


def create_ensemble(name: str, components: list, repl_dir: Path, seed: int = 42):
    """
    Combina sintéticos de múltiples configs.

    Args:
        name: Nombre del ensemble
        components: Lista de nombres de configs
        repl_dir: Directorio con los resultados
        seed: Seed usado en generación
    """
    print(f"\nCreando {name}...")

    dfs = []
    for config in components:
        path = repl_dir / f"{config}_s{seed}_synth.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['source'] = config
            dfs.append(df)
            print(f"  {config}: {len(df)} sintéticos")
        else:
            print(f"  WARNING: {path} no encontrado")

    if not dfs:
        print(f"  ERROR: No se encontraron sintéticos para {name}")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    before_dedup = len(combined)

    # Deduplicar por texto
    combined = combined.drop_duplicates(subset=['text'])
    after_dedup = len(combined)

    dedup_removed = before_dedup - after_dedup
    if dedup_removed > 0:
        print(f"  Deduplicados: {dedup_removed} removidos")

    output_path = repl_dir / f"{name}_synth.csv"
    combined.to_csv(output_path, index=False)
    print(f"  -> {name}: {len(combined)} sintéticos totales")

    return combined


def main():
    parser = argparse.ArgumentParser(description='Create ensembles from replication')
    parser.add_argument('--repl-dir', type=str, required=True, help='Replication results directory')
    parser.add_argument('--seed', type=int, default=42, help='Seed used in generation')
    args = parser.parse_args()

    repl_dir = Path(args.repl_dir)

    print("=" * 70)
    print("Creating Ensembles from Replication")
    print("=" * 70)
    print(f"Directory: {repl_dir}")

    # ENS_Top3_G5 (base ensemble)
    create_ensemble("ENS_Top3_G5", [
        "CMB3_skip",
        "CF1_conf_band",
        "V4_ultra",
        "G5_K25_medium"
    ], repl_dir, args.seed)

    # ENS_TopG5_Extended (our best)
    create_ensemble("ENS_TopG5_Extended", [
        "CMB3_skip",
        "CF1_conf_band",
        "V4_ultra",
        "G5_K25_medium",
        "W9_contrastive",
        "W1_low_gate"
    ], repl_dir, args.seed)

    # ENS_SUPER_G5_F7_v2
    create_ensemble("ENS_SUPER_G5_F7_v2", [
        "CMB3_skip",
        "CF1_conf_band",
        "V4_ultra",
        "G5_K25_medium",
        "W1_force_problem",
        "EXP7_hybrid_best",
        "W3_no_dedup"
    ], repl_dir, args.seed)

    print("\n" + "=" * 70)
    print("Ensembles created successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
