import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

print("="*60)
print("SPANISH BERT EMBEDDINGS FOR NEUTRAL AND NEGATIVE REVIEWS")
print("="*60)

# Load the stratified sample
print("\n1. Loading stratified sample...")
sample_df = pd.read_pickle("../../data/Madrid_stratified_sample_10000.pkl")
print(f"Sample loaded: {len(sample_df)} reviews")
print("Sentiment distribution:")
print(sample_df['sentiment'].value_counts())

# Filter for neutral and negative reviews only
neutral_negative_df = sample_df[sample_df['sentiment'].isin(['neutral', 'negative'])].copy().reset_index(drop=True)
print(f"\nFiltered dataset: {len(neutral_negative_df)} reviews (neutral + negative)")
print("Distribution:")
print(neutral_negative_df['sentiment'].value_counts())

# Load Spanish BERT model
print("\n2. Loading Spanish BERT model...")
# Using 'dccuchile/bert-base-spanish-wwm-uncased' - a popular Spanish BERT model
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f"✓ Model loaded successfully: {model_name}")
    print(f"✓ Tokenizer vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"Failed to load {model_name}, trying alternative...")
    # Alternative: Spanish BERT from BSC
    model_name = 'PlanTL-GOB-ES/roberta-base-bne'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f"✓ Alternative model loaded: {model_name}")

# Set model to evaluation mode
model.eval()

# Function to get BERT embeddings
def get_bert_embeddings(texts, batch_size=8):
    """
    Get BERT embeddings for a list of texts
    Returns the [CLS] token embeddings (768-dimensional)
    """
    embeddings = []
    
    print(f"\n3. Generating embeddings for {len(texts)} texts...")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# Prepare texts for embedding
texts_to_embed = neutral_negative_df['text_clean'].tolist()
print(f"\nTexts prepared for embedding: {len(texts_to_embed)}")

# Generate embeddings
embeddings = get_bert_embeddings(texts_to_embed, batch_size=8)

print(f"\n4. Embeddings generated!")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Add embeddings to the dataframe
print(f"\n5. Saving results...")

# Save embeddings as separate numpy file
embeddings_file = "../../data/Madrid_neutral_negative_bert_embeddings.npy"
np.save(embeddings_file, embeddings)
print(f"✓ Embeddings saved to: {embeddings_file}")

# Save the filtered dataset with metadata
filtered_dataset_file = "../../data/Madrid_neutral_negative_sample.pkl"
neutral_negative_df.to_pickle(filtered_dataset_file)
print(f"✓ Filtered dataset saved to: {filtered_dataset_file}")

# Also save as CSV
csv_file = "../../data/Madrid_neutral_negative_sample.csv"
neutral_negative_df.to_csv(csv_file, index=False)
print(f"✓ Dataset also saved as CSV: {csv_file}")

# Create a summary
print(f"\n6. SUMMARY")
print("="*40)
print(f"Model used: {model_name}")
print(f"Total reviews embedded: {len(neutral_negative_df)}")
print(f"Negative reviews: {len(neutral_negative_df[neutral_negative_df['sentiment'] == 'negative'])}")
print(f"Neutral reviews: {len(neutral_negative_df[neutral_negative_df['sentiment'] == 'neutral'])}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Total parameters embedded: {embeddings.size:,}")

# Show some statistics about the embeddings
print(f"\nEmbedding statistics:")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std: {embeddings.std():.4f}")
print(f"  Min: {embeddings.min():.4f}")
print(f"  Max: {embeddings.max():.4f}")

# Save embedding metadata
metadata = {
    'model_name': model_name,
    'embedding_shape': embeddings.shape,
    'embedding_dimension': embeddings.shape[1],
    'num_negative_reviews': len(neutral_negative_df[neutral_negative_df['sentiment'] == 'negative']),
    'num_neutral_reviews': len(neutral_negative_df[neutral_negative_df['sentiment'] == 'neutral']),
    'total_reviews': len(neutral_negative_df),
    'files_created': [embeddings_file, filtered_dataset_file, csv_file]
}

import json
metadata_file = "../../data/Madrid_bert_embeddings_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"✓ Metadata saved to: {metadata_file}")

print(f"\n✓ BERT EMBEDDING PROCESS COMPLETED!")
print("="*60)