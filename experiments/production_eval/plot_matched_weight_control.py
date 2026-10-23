"""Standalone scientific figure from completed matched-estimator results."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/production_eval_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.production_eval.matched_weight_control import BASE


def main():
    output = BASE / "outcome_aware/matched_weight_control"
    result = json.loads((output / "results.json").read_text())
    rows = result["exact_conditional_design_moments"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, dataset, budget, title in zip(axes, ("bbh_boolean_expressions", "gsm8k"), (25, 50),
                                           ("Lógica booleana · 25 familias", "GSM8K · 50 familias")):
        scoped = [r for r in rows if r["dataset"] == dataset and r["budget_families"] == budget and r["target"] == "construction"]
        pairs = [("qwen_direct", "qwen_work"), ("qwen_work", "gemini_direct")]
        for offset, signal, estimator, label, color in [
            (-.24, "error", "srs", "Aleatorio uniforme", "#637180"),
            (0., "error", "ht", "Fallos + aleatorio, HT", "#397ba6"),
            (.24, "disagreement", "ht", "Desacuerdos + aleatorio, HT", "#249b88"),
        ]:
            values = [100 * next(r for r in scoped if r["risk_signal"] == signal and
                                 (r["anchor_config"], r["candidate_config"]) == pair)[estimator]["rmse_to_target"] for pair in pairs]
            bars = axis.bar(np.arange(2) + offset, values, width=.22, color=color, label=label)
            axis.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9)
        axis.set_xticks([0, 1], ["Qwen pasos − directo\nPar usado en desacuerdo", "Gemini − Qwen pasos\nGemini no usado al seleccionar"])
        axis.set_title(title, loc="left", fontweight="bold", pad=17)
        axis.set_ylim(0, 10)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("RMSE exacto de muestreo (puntos porcentuales)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    figure.text(.075, .045, "Condicionado al pool de construcción observado y a las cuotas fijas. Sin incertidumbre de modelo o población futura.", fontsize=9, color="#4c5560")
    figure.subplots_adjust(top=.81, bottom=.24, left=.075, right=.985, wspace=.18)
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output / ("transfer_precision." + extension), dpi=180)
    plt.close(figure)
    print(output / "transfer_precision.png")


if __name__ == "__main__":
    main()
