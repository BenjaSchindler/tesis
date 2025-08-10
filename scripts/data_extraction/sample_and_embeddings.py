import pandas as pd
import numpy as np

# Load the Spanish reviews sentiment dataset
print("Loading Spanish reviews dataset...")
spanish_reviews = pd.read_pickle("../../data/Madrid_spanish_reviews_sentiment.pkl")

print(f"Dataset loaded: {len(spanish_reviews):,} reviews")
print("\nSentiment distribution:")
sentiment_counts = spanish_reviews['sentiment'].value_counts()
print(sentiment_counts)
print("\nProportions:")
proportions = sentiment_counts / len(spanish_reviews)
print(proportions.round(4))

# Create stratified sample of 1000 reviews maintaining proportions
print("\n" + "="*50)
print("CREATING STRATIFIED SAMPLE")
print("="*50)

# Calculate target counts for each class to maintain proportions
target_sample_size = 10000
sample_counts = {}
for sentiment in sentiment_counts.index:
    sample_counts[sentiment] = int(proportions[sentiment] * target_sample_size)

# Adjust to ensure exactly 1000 samples (handle rounding)
total_calculated = sum(sample_counts.values())
if total_calculated != target_sample_size:
    # Add the difference to the largest class
    largest_class = sentiment_counts.index[0]
    sample_counts[largest_class] += (target_sample_size - total_calculated)

print("Target sample sizes:")
for sentiment, count in sample_counts.items():
    print(f"  {sentiment}: {count} samples ({count/target_sample_size*100:.1f}%)")

# Create stratified sample
sample_dfs = []
for sentiment, target_count in sample_counts.items():
    sentiment_data = spanish_reviews[spanish_reviews['sentiment'] == sentiment]
    if len(sentiment_data) >= target_count:
        sample = sentiment_data.sample(n=target_count, random_state=42)
    else:
        sample = sentiment_data  # Use all if not enough samples
    sample_dfs.append(sample)

# Combine samples
stratified_sample = pd.concat(sample_dfs, ignore_index=True)

print(f"\nActual sample created: {len(stratified_sample)} reviews")
print("Actual sample distribution:")
sample_sentiment_counts = stratified_sample['sentiment'].value_counts()
print(sample_sentiment_counts)
print("\nSample proportions:")
sample_proportions = sample_sentiment_counts / len(stratified_sample)
print(sample_proportions.round(4))

# Save the stratified sample
sample_file = "../../data/Madrid_stratified_sample_10000.pkl"
stratified_sample.to_pickle(sample_file)
print(f"\nStratified sample saved to: {sample_file}")

# Also save as CSV for easy inspection
csv_file = "../../data/Madrid_stratified_sample_10000.csv"
stratified_sample.to_csv(csv_file, index=False)
print(f"Sample also saved as CSV: {csv_file}")

print("\n" + "="*50)
print("SAMPLE CREATION COMPLETED")
print("="*50)