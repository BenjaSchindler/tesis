# Data Organization Summary

## Project Structure

```
Tesis/
├── CLAUDE.md                           # Project instructions for Claude
├── README_Data_Organization.md          # This file
├── 
├── # Jupyter Notebooks
├── Data_Extrction.ipynb                # Original data exploration notebook
├── Sentiment_Analysis_Spanish_Reviews.ipynb # Sentiment analysis notebook
├── 
├── # Python Scripts
├── sample_and_embeddings.py            # Creates stratified samples
├── bert_embeddings.py                  # Generates BERT embeddings
├── smote_balancing.py                  # SMOTE data balancing
├── llm_balancing.py                    # GPT-5 Mini data balancing
├── llm_balancing_demo.py               # Demo version (no API calls)
├── setup_llm_generation.py             # LLM setup with API key
├── 
└── data/                               # All data files organized here
    ├── Madrid.zip                      # Original compressed dataset
    ├── Madrid/                         # Original dataset folder
    │   ├── items.pkl                   # Restaurant/item data
    │   ├── reviews.pkl                 # Raw review data (1.3M reviews)
    │   └── users.pkl                   # User data
    ├── 
    ├── # Processed Datasets
    ├── Madrid_spanish_reviews_sentiment.pkl    # Spanish reviews with sentiment labels
    ├── Madrid_spanish_reviews_sentiment.csv    # CSV version
    ├── Madrid_stratified_sample_10000.pkl      # 10K stratified sample
    ├── Madrid_stratified_sample_10000.csv      # CSV version
    ├── 
    ├── # BERT Embeddings
    ├── Madrid_neutral_negative_sample.pkl      # Filtered neutral/negative reviews
    ├── Madrid_neutral_negative_sample.csv      # CSV version
    ├── Madrid_neutral_negative_bert_embeddings.npy  # BERT embeddings (2683×768)
    ├── Madrid_bert_embeddings_metadata.json    # BERT embedding metadata
    ├── 
    ├── # SMOTE Balanced Data
    ├── Madrid_balanced_smote_dataset.pkl       # SMOTE balanced dataset
    ├── Madrid_balanced_smote_dataset.csv       # CSV version
    ├── Madrid_smote_embeddings.npy             # SMOTE resampled embeddings
    ├── Madrid_synthetic_smote_samples.pkl      # Synthetic SMOTE samples only
    ├── Madrid_smote_metadata.json              # SMOTE metadata
    ├── Madrid_SMOTE_analysis.png               # SMOTE analysis visualization
    ├── 
    ├── # LLM Demo Data
    ├── Madrid_llm_generation_demo.pkl          # Demo generated samples
    └── Madrid_llm_demo_metadata.json           # Demo metadata
```

## Dataset Sizes

| Dataset | Samples | Purpose |
|---------|---------|---------|
| Original (reviews.pkl) | 1,347,959 | Raw TripAdvisor reviews |
| Spanish sentiment | 964,056 | Spanish reviews with labels |
| 10K stratified sample | 10,000 | Proportional sample (73.2% pos, 14.8% neg, 12.1% neu) |
| Neutral/Negative subset | 2,683 | For BERT embedding |
| SMOTE balanced | 21,951 | Perfectly balanced (7,317 each class) |

## Balancing Methods Comparison

### SMOTE Approach
- **Method**: Synthetic Minority Oversampling Technique
- **Input**: BERT embeddings (768-dimensional)
- **Output**: 21,951 samples (11,951 synthetic + 10,000 original)
- **Cost**: Computational only
- **Quality**: Mathematical interpolation in embedding space

### GPT-5 Mini Approach
- **Method**: Large Language Model generation with RAG
- **Input**: Similar reviews as context
- **Output**: 21,951 samples (11,951 LLM generated + 10,000 original)  
- **Cost**: ~$3.67 USD (API calls)
- **Quality**: Authentic Spanish text generation

## Key Files for Analysis

1. **data/Madrid_stratified_sample_10000.pkl** - Main working dataset
2. **data/Madrid_balanced_smote_dataset.pkl** - SMOTE balanced version
3. **data/Madrid_neutral_negative_bert_embeddings.npy** - BERT embeddings for minority classes
4. **data/Madrid_smote_embeddings.npy** - SMOTE generated embeddings

## Usage Notes

- All scripts now reference the `data/` folder for file paths
- Old 1000-sample files have been removed
- Current sample size is 10,000 reviews (not 1,000)
- BERT embeddings are 768-dimensional vectors
- SMOTE balancing creates perfect 1:1:1 class distribution

## Next Steps

To generate LLM-balanced data:
```bash
python setup_llm_generation.py
```

To test with demo (no API calls):
```bash
python llm_balancing_demo.py
```