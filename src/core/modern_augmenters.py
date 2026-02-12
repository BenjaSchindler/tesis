#!/usr/bin/env python3
"""
Modern Augmentation Baselines for Text Classification

Three modern augmentation methods to compare against geometric filtering:
  1. EmbeddingMixupAugmenter — interpolation in embedding space (MixText-inspired)
  2. T5ParaphraseAugmenter  — paraphrasing with local T5 model
  3. ContextualBERTAugmenter — contextual word substitution with BERT MLM
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================================
# EMBEDDING MIXUP (operates directly in embedding space)
# ============================================================================

class EmbeddingMixupAugmenter:
    """
    Generates synthetic samples by interpolating pairs of real embeddings.

    synth = alpha * emb1 + (1 - alpha) * emb2
    where alpha ~ Beta(beta_a, beta_b)

    Inspired by MixText (Chen et al., 2020) but applied to sentence embeddings.
    """

    def __init__(self, beta_a=0.4, beta_b=0.4, random_state=42):
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.rng = np.random.RandomState(random_state)

    def generate_for_class(self, real_embeddings, n_generate):
        """Generate n synthetic embeddings by mixing pairs from real_embeddings."""
        n_real = len(real_embeddings)
        if n_real < 2:
            # Can't mix with fewer than 2 samples — duplicate with noise
            if n_real == 1:
                noise = self.rng.normal(0, 0.01, (n_generate, real_embeddings.shape[1]))
                return real_embeddings[0] + noise
            return np.array([]).reshape(0, real_embeddings.shape[1])

        # Sample random pairs
        idx1 = self.rng.randint(0, n_real, size=n_generate)
        idx2 = self.rng.randint(0, n_real, size=n_generate)
        # Ensure pairs are different
        same_mask = idx1 == idx2
        idx2[same_mask] = (idx2[same_mask] + 1) % n_real

        # Sample interpolation coefficients
        alphas = self.rng.beta(self.beta_a, self.beta_b, size=n_generate).reshape(-1, 1)

        # Interpolate
        synthetic = alphas * real_embeddings[idx1] + (1 - alphas) * real_embeddings[idx2]
        return synthetic


# ============================================================================
# T5 PARAPHRASE (local model, cached)
# ============================================================================

class T5ParaphraseAugmenter:
    """
    Paraphrases training texts using a local T5 model fine-tuned for paraphrasing.

    Uses Vamsi/T5_Paraphrase_Paws by default (~220MB).
    Results are cached to disk using MD5 hashing.
    """

    def __init__(self, model_name="Vamsi/T5_Paraphrase_Paws", cache_dir=None,
                 device=None, max_length=256, num_beams=5, num_return_sequences=1):
        import torch
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"    Loading T5 paraphrase model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def _get_cache_path(self, text):
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return self.cache_dir / f"t5_{text_hash}.json"

    def paraphrase(self, text):
        """Paraphrase a single text, using cache if available."""
        cache_path = self._get_cache_path(text)
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                return json.load(f).get("paraphrased", text)

        self._load_model()

        input_text = f"paraphrase: {text} </s>"
        encoding = self._tokenizer(
            input_text, return_tensors="pt",
            max_length=self.max_length, truncation=True, padding=True,
        ).to(self.device)

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **encoding,
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                early_stopping=True,
                do_sample=True,
                temperature=0.9,
                top_k=50,
            )
        paraphrased = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cache result
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump({
                    "original": text,
                    "paraphrased": paraphrased,
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)

        return paraphrased

    def _paraphrase_batch(self, texts, batch_size=32):
        """Paraphrase a list of texts with batched GPU inference. Respects cache."""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if cache_path and cache_path.exists():
                with open(cache_path) as f:
                    results[i] = json.load(f).get("paraphrased", text)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results

        self._load_model()
        import torch

        # Process uncached texts in batches
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start:batch_start + batch_size]
            input_texts = [f"paraphrase: {t} </s>" for t in batch]

            encoding = self._tokenizer(
                input_texts, return_tensors="pt",
                max_length=self.max_length, truncation=True, padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **encoding,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_return_sequences,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                )

            for j, output in enumerate(outputs):
                global_idx = uncached_indices[batch_start + j]
                paraphrased = self._tokenizer.decode(output, skip_special_tokens=True)
                results[global_idx] = paraphrased

                # Cache result
                cache_path = self._get_cache_path(batch[j])
                if cache_path:
                    with open(cache_path, "w") as f:
                        json.dump({
                            "original": batch[j],
                            "paraphrased": paraphrased,
                            "model": self.model_name,
                            "timestamp": datetime.now().isoformat(),
                        }, f, indent=2)

        return results

    def generate_for_class(self, texts, n_generate, embed_model):
        """Generate n_generate paraphrases from the given texts, then embed them."""
        # Cycle through source texts to reach n_generate
        source = texts * (n_generate // len(texts) + 1)
        source = source[:n_generate]
        paraphrased_texts = self._paraphrase_batch(source)

        embeddings = embed_model.encode(paraphrased_texts, show_progress_bar=False)
        return embeddings, paraphrased_texts


# ============================================================================
# CONTEXTUAL BERT AUGMENTATION (MLM-based word substitution)
# ============================================================================

class ContextualBERTAugmenter:
    """
    Augments text by masking random tokens and filling them with BERT MLM predictions.

    More semantically coherent than EDA (Wei & Zou 2019) because substitutions
    are context-aware rather than random.
    """

    def __init__(self, model_name="bert-base-uncased", mask_prob=0.15,
                 device=None, random_state=42):
        import torch
        self.model_name = model_name
        self.mask_prob = mask_prob
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.RandomState(random_state)
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            print(f"    Loading BERT MLM model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def augment(self, text):
        """Augment a single text by masking and filling tokens."""
        self._load_model()
        import torch

        tokens = self._tokenizer.tokenize(text)
        if len(tokens) < 3:
            return text

        # Determine which tokens to mask (skip special tokens)
        n_mask = max(1, int(len(tokens) * self.mask_prob))
        mask_indices = self.rng.choice(len(tokens), size=min(n_mask, len(tokens)), replace=False)

        # Create masked version
        masked_tokens = tokens.copy()
        for idx in mask_indices:
            masked_tokens[idx] = self._tokenizer.mask_token

        masked_text = self._tokenizer.convert_tokens_to_string(masked_tokens)
        encoding = self._tokenizer(
            masked_text, return_tensors="pt",
            max_length=512, truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encoding)
            logits = outputs.logits

        # Find mask token positions in the encoded input
        input_ids = encoding["input_ids"][0]
        mask_token_id = self._tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        # Replace masks with top predictions
        result_ids = input_ids.clone()
        for pos in mask_positions:
            predicted_id = logits[0, pos].argmax().item()
            result_ids[pos] = predicted_id

        # Decode back to text
        augmented = self._tokenizer.decode(result_ids, skip_special_tokens=True)
        return augmented

    def _augment_batch(self, texts, batch_size=64):
        """Augment a list of texts with batched GPU inference."""
        self._load_model()
        import torch

        # Pre-process: tokenize and mask all texts
        masked_texts = []
        too_short = {}  # index -> original text (skip masking)
        for i, text in enumerate(texts):
            tokens = self._tokenizer.tokenize(text)
            if len(tokens) < 3:
                too_short[i] = text
                masked_texts.append(text)  # placeholder
                continue
            n_mask = max(1, int(len(tokens) * self.mask_prob))
            mask_indices = self.rng.choice(len(tokens), size=min(n_mask, len(tokens)), replace=False)
            masked_tokens = tokens.copy()
            for idx in mask_indices:
                masked_tokens[idx] = self._tokenizer.mask_token
            masked_texts.append(self._tokenizer.convert_tokens_to_string(masked_tokens))

        # Batch inference
        results = [None] * len(texts)
        # Fill in too-short texts
        for i, text in too_short.items():
            results[i] = text

        # Process maskable texts in batches
        processable = [(i, masked_texts[i]) for i in range(len(texts)) if i not in too_short]
        for batch_start in range(0, len(processable), batch_size):
            batch = processable[batch_start:batch_start + batch_size]
            batch_indices = [b[0] for b in batch]
            batch_texts = [b[1] for b in batch]

            encoding = self._tokenizer(
                batch_texts, return_tensors="pt",
                max_length=512, truncation=True, padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**encoding)
                logits = outputs.logits

            mask_token_id = self._tokenizer.mask_token_id
            for j, global_idx in enumerate(batch_indices):
                input_ids = encoding["input_ids"][j]
                mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
                result_ids = input_ids.clone()
                for pos in mask_positions:
                    predicted_id = logits[j, pos].argmax().item()
                    result_ids[pos] = predicted_id
                results[global_idx] = self._tokenizer.decode(result_ids, skip_special_tokens=True)

        return results

    def generate_for_class(self, texts, n_generate, embed_model):
        """Generate n_generate augmented texts, then embed them."""
        # Cycle through source texts, augmenting each potentially multiple times
        source = texts * (n_generate // len(texts) + 1)
        source = source[:n_generate]
        augmented_texts = self._augment_batch(source)

        embeddings = embed_model.encode(augmented_texts, show_progress_bar=False)
        return embeddings, augmented_texts
