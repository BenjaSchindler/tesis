"""Split the paper's t-SNE figure into its two panels for the talk.

The panel titles and the legend of the source PNG are too small to read on a
projector, so they are cropped away and redrawn in LaTeX (slide 9).
Usage: python figures/crop_tsne.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SRC = HERE / "fig_tsne_before_after_emotion_en.png"

# Pixel bands of the 5339x2241 source: panel titles end at row 150, the
# x-axis labels end at row 1995 (legend below), panels split at column 2660.
TOP, BOTTOM, SPLIT = 150, 1995, 2660

im = Image.open(SRC)
im.crop((0, TOP, SPLIT, BOTTOM)).save(HERE / "fig_tsne_emotion_before.png", optimize=True)
im.crop((SPLIT, TOP, im.width, BOTTOM)).save(HERE / "fig_tsne_emotion_after.png", optimize=True)
