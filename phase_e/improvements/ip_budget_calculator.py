#!/usr/bin/env python3
"""
Improvement Potential (IP) Budget Calculator

Métrica descubierta en análisis Phase E:
  IP = 1 - baseline_f1

Hallazgo:
  - Clases con IP > 0.7: +2.08% delta promedio
  - Clases con IP <= 0.7: -1.52% delta promedio

Esta implementación usa IP como multiplicador continuo para el presupuesto:
  ip_mult = 1 + ip_boost_factor * IP

Donde ip_boost_factor controla cuánto extra presupuesto dar a clases con bajo F1.
"""

from typing import Dict, Tuple


def calculate_ip_enhanced_budget(
    n_samples: int,
    quality_score: float,
    purity: float,
    baseline_f1: float,
    # IP parameters (NEW)
    use_ip_scaling: bool = True,
    ip_boost_factor: float = 1.0,  # Range: 0.5-2.0 recommended
    ip_threshold: float = 0.7,     # Only apply IP boost if IP > threshold
    # Legacy parameters
    purity_low_threshold: float = 0.30,
    purity_low_multiplier: float = 0.3,
    target_ratio: float = 0.08
) -> Tuple[int, str, Dict[str, float]]:
    """
    Enhanced budget calculator with continuous IP scaling.

    Formula:
        budget = base_budget × quality_mult × purity_mult × ip_mult

    Where:
        IP = 1 - baseline_f1  (Improvement Potential)
        ip_mult = 1 + ip_boost_factor × IP  (if IP > ip_threshold)

    Args:
        n_samples: Number of real samples in class
        quality_score: Anchor quality score (0-1)
        purity: Anchor purity score (0-1)
        baseline_f1: Baseline F1 before augmentation
        use_ip_scaling: Whether to apply IP-based scaling
        ip_boost_factor: How much to boost budget based on IP (1.0 = double for IP=1)
        ip_threshold: Only apply boost if IP > threshold
        purity_low_threshold: Purity below which to apply reduction
        purity_low_multiplier: Multiplier for low purity
        target_ratio: Target synthetic/real ratio

    Returns:
        budget: Number of synthetics to generate
        reason: Explanation of budget calculation
        multipliers: Dict with individual multipliers

    Example:
        For baseline_f1 = 0.25:
        - IP = 1 - 0.25 = 0.75
        - ip_mult = 1 + 1.0 × 0.75 = 1.75
        - Budget gets 75% boost!

        For baseline_f1 = 0.80:
        - IP = 1 - 0.80 = 0.20 (below threshold)
        - ip_mult = 1.0 (no boost, class already performs well)
    """
    # Base budget: target_ratio% of real samples
    base_budget = int(n_samples * target_ratio)

    # 1. Quality multiplier (from Phase 1)
    if quality_score < 0.35:
        quality_mult = 0.1
        quality_reason = f"🔴 Very low quality ({quality_score:.3f})"
    elif quality_score < 0.40:
        quality_mult = 0.3
        quality_reason = f"⚠️  Low quality ({quality_score:.3f})"
    elif quality_score < 0.50:
        quality_mult = 0.7
        quality_reason = f"⚠️  Mediocre quality ({quality_score:.3f})"
    else:
        quality_mult = 1.0
        quality_reason = f"✅ Good quality ({quality_score:.3f})"

    # 2. Purity multiplier (from Phase 2)
    if purity < purity_low_threshold:
        purity_mult = purity_low_multiplier
        purity_reason = f"🔴 Low purity ({purity:.3f}) → {int(purity_mult*100)}%"
    else:
        purity_mult = 1.0
        purity_reason = f"✅ Purity OK ({purity:.3f})"

    # 3. IP multiplier (NEW - Phase E)
    ip = 1 - baseline_f1  # Improvement Potential

    if use_ip_scaling and ip > ip_threshold:
        # Continuous scaling: more IP = more budget
        ip_mult = 1 + ip_boost_factor * ip
        ip_reason = f"📈 IP={ip:.3f} (high potential) → {int(ip_mult*100)}% boost"
    elif use_ip_scaling and ip > 0.5:
        # Moderate IP: small boost
        ip_mult = 1 + 0.5 * ip_boost_factor * ip
        ip_reason = f"📊 IP={ip:.3f} (moderate) → {int(ip_mult*100)}%"
    else:
        ip_mult = 1.0
        ip_reason = f"✅ IP={ip:.3f} (low potential, no boost needed)"

    # Combine multipliers
    total_mult = quality_mult * purity_mult * ip_mult
    budget = int(base_budget * total_mult)

    # IP RESCUE: For high-IP classes, ensure minimum budget scales with IP
    # This prevents quality/purity from completely blocking augmentation
    # for classes that really need it
    if use_ip_scaling and ip > ip_threshold:
        ip_minimum = int(10 + 20 * ip)  # Range: 17-30 for IP 0.7-1.0
        if budget < ip_minimum:
            budget = ip_minimum
            ip_reason += f" → IP rescue: min {ip_minimum}"

    budget = max(10, budget)

    # Build reason string
    reason = f"Base: {base_budget} ({int(target_ratio*100)}% of {n_samples})\n"
    reason += f"   {quality_reason} × {int(quality_mult*100)}%\n"
    reason += f"   {purity_reason} × {int(purity_mult*100)}%\n"
    reason += f"   {ip_reason} × {int(ip_mult*100)}%\n"
    reason += f"   → Final: {budget} synthetics (×{total_mult:.2f})"

    multipliers = {
        "quality": quality_mult,
        "purity": purity_mult,
        "ip": ip_mult,
        "improvement_potential": ip,
        "total": total_mult
    }

    return budget, reason, multipliers


def demo():
    """Demonstrate IP budget calculation for different scenarios."""
    print("=" * 70)
    print("  DEMO: IP-Enhanced Budget Calculator")
    print("=" * 70)

    # Test cases based on Phase E analysis
    test_cases = [
        # (class_name, n_samples, quality, purity, baseline_f1)
        ("ISFJ (worst baseline)", 130, 0.306, 0.016, 0.252),
        ("ISTJ", 249, 0.308, 0.025, 0.235),
        ("ESFJ", 36, 0.302, 0.009, 0.222),
        ("ESFP", 72, 0.323, 0.041, 0.344),
        ("ESTP (best baseline)", 397, 0.528, 0.570, 0.791),
        ("ESTJ", 96, 0.389, 0.273, 0.606),
    ]

    print(f"\n{'Clase':<25} {'IP':<8} {'Budget (old)':<14} {'Budget (IP)':<14} {'Boost':<10}")
    print("-" * 75)

    for name, n, q, p, f1 in test_cases:
        # Old method (no IP)
        old_budget, _, _ = calculate_ip_enhanced_budget(
            n, q, p, f1, use_ip_scaling=False
        )

        # New method (with IP)
        new_budget, reason, mults = calculate_ip_enhanced_budget(
            n, q, p, f1, use_ip_scaling=True, ip_boost_factor=1.0
        )

        ip = mults['improvement_potential']
        boost = (new_budget - old_budget) / old_budget * 100 if old_budget > 0 else 0

        print(f"{name:<25} {ip:<8.3f} {old_budget:<14} {new_budget:<14} {boost:>+8.1f}%")

    print("\n" + "=" * 70)
    print("  Ejemplo detallado: ISFJ")
    print("=" * 70)

    budget, reason, mults = calculate_ip_enhanced_budget(
        130, 0.306, 0.016, 0.252,
        use_ip_scaling=True, ip_boost_factor=1.0
    )
    print(f"\n{reason}")


if __name__ == "__main__":
    demo()
