"""Download one pinned official trainable Qwen checkpoint, without remote code."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval import download_models_v2 as downloader


def main():
    downloader.OUT = ROOT / "output/research/production_eval/experiments_v3/models"
    downloader.SPECS = [(
        "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "LICENSE", "README.md", "model.safetensors"],
    )]
    # The reusable downloader resolves this exact revision, verifies upstream
    # LFS SHA256 and exact sizes, streams to a temporary file, and records all
    # assets. No model loader or downloaded executable code is invoked.
    downloader.main()


if __name__ == "__main__":
    main()
