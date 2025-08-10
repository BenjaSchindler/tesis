# Project Structure - Thesis Data Balancing

## Overview

This project focuses on balancing Spanish restaurant review data for sentiment analysis. The scripts are organized by functionality to support a clean data science workflow.

## Directory Structure

```
Tesis/
├── CLAUDE.md                           # Project instructions for Claude
├── README_Data_Organization.md          # Data files documentation  
├── README_Project_Structure.md          # This file - project structure guide
├── 
├── data/                               # All datasets and processed files
│   ├── Madrid.zip                      # Original compressed dataset
│   ├── Madrid/                         # Raw data folder
│   │   ├── items.pkl                   # Restaurant/item data (1.3M reviews)
│   │   ├── reviews.pkl                 # Raw review data
│   │   └── users.pkl                   # User data
│   │
│   ├── # Processed Datasets
│   ├── Madrid_spanish_reviews_sentiment.pkl/csv    # Spanish reviews with sentiment labels (964K)
│   ├── Madrid_stratified_sample_10000.pkl/csv      # Stratified sample (10K reviews)
│   │
│   ├── # BERT Embeddings  
│   ├── Madrid_neutral_negative_sample.pkl/csv      # Filtered neutral/negative (2.7K)
│   ├── Madrid_neutral_negative_bert_embeddings.npy # BERT vectors (2683×768)
│   ├── Madrid_bert_embeddings_metadata.json        # Embedding metadata
│   │
│   ├── # SMOTE Balanced Data
│   ├── Madrid_balanced_smote_dataset.pkl/csv       # SMOTE balanced (21.9K samples)
│   ├── Madrid_smote_embeddings.npy                 # SMOTE embeddings
│   ├── Madrid_synthetic_smote_samples.pkl          # Synthetic samples only
│   ├── Madrid_smote_metadata.json                  # SMOTE parameters & results
│   ├── Madrid_SMOTE_analysis.png                   # SMOTE visualization
│   │
│   └── # LLM Generated Data
│       ├── Madrid_llm_generation_demo.pkl          # Demo generated samples
│       └── Madrid_llm_demo_metadata.json           # Demo metadata
│
└── scripts/                            # Organized by functionality
    ├── data_exploration/               # Data analysis and visualization
    │   ├── Data_Extrction.ipynb       # Comprehensive data exploration (1.3M reviews)
    │   └── Sentiment_Analysis_Spanish_Reviews.ipynb # Sentiment analysis & class imbalance
    │
    ├── data_extraction/               # Data preprocessing and feature extraction  
    │   ├── sample_and_embeddings.py   # Create stratified samples
    │   └── bert_embeddings.py         # Generate BERT embeddings
    │
    ├── data_augmentation/             # Data balancing techniques
    │   ├── smote_balancing.py         # SMOTE oversampling
    │   ├── llm_balancing.py           # GPT-5 Mini generation
    │   ├── llm_balancing_demo.py      # LLM demo (no API calls)
    │   └── setup_llm_generation.py    # LLM setup and API key management
    │
    └── model_training/                # Machine learning models (empty - ready for ML scripts)
```

## Workflow Pipeline

### 1. Data Exploration (`scripts/data_exploration/`)

**Purpose**: Understand the dataset characteristics and class imbalance

- **Data_Extrction.ipynb**: Comprehensive analysis of 1.3M Spanish restaurant reviews
  - Text length analysis, language distribution
  - Rating patterns, temporal analysis
  - Data quality assessment
  
- **Sentiment_Analysis_Spanish_Reviews.ipynb**: Focus on class imbalance
  - Sentiment labeling (negative: 10-20, neutral: 30, positive: 40-50)
  - **Severe imbalance**: 73.2% positive, 14.8% negative, 12.1% neutral
  - **Imbalance ratio**: 6.05:1 (extreme classification)

### 2. Data Extraction (`scripts/data_extraction/`)

**Purpose**: Create working datasets and embeddings

- **sample_and_embeddings.py**: Create stratified samples
  - Generates proportional 10K sample from 964K Spanish reviews
  - Maintains original class distribution
  - Exports: `Madrid_stratified_sample_10000.pkl/csv`

- **bert_embeddings.py**: Generate BERT embeddings  
  - Uses `dccuchile/bert-base-spanish-wwm-uncased`
  - Embeds 2,683 neutral/negative reviews → 768-dimensional vectors
  - Exports: `Madrid_neutral_negative_bert_embeddings.npy`

### 3. Data Augmentation (`scripts/data_augmentation/`)

**Purpose**: Balance the dataset using different techniques

#### SMOTE Approach
- **smote_balancing.py**: Statistical oversampling
  - Input: BERT embeddings (768D space)
  - Method: k-neighbors interpolation (k=5)
  - Output: 21,951 balanced samples (7,317 per class)
  - **Cost**: Computational only
  - **Quality**: Mathematical interpolation

#### LLM Approach  
- **llm_balancing.py**: Authentic text generation
  - Model: GPT-5 Mini with RAG system
  - Method: Context-aware Spanish review generation
  - Output: 21,951 balanced samples (11,951 LLM generated)
  - **Cost**: ~$3.67 USD (API calls)
  - **Quality**: Authentic Spanish text with varied lengths

- **llm_balancing_demo.py**: Cost-free demonstration
- **setup_llm_generation.py**: API key setup and cost estimation

### 4. Model Training (`scripts/model_training/`)

**Purpose**: Ready for ML model implementation

*Currently empty - prepared for:*
- Classification model training scripts
- Performance evaluation and comparison
- Cross-validation and hyperparameter tuning
- Results analysis and visualization

## Usage Instructions

### Running the Complete Pipeline

1. **Data Exploration** (optional - for understanding):
   ```bash
   # Open Jupyter notebooks in scripts/data_exploration/
   jupyter notebook scripts/data_exploration/
   ```

2. **Data Extraction** (required - creates working datasets):
   ```bash
   cd scripts/data_extraction/
   python sample_and_embeddings.py    # Creates 10K sample
   python bert_embeddings.py          # Creates BERT embeddings
   ```

3. **Data Augmentation** (choose method):
   
   **Option A - SMOTE Balancing:**
   ```bash
   cd scripts/data_augmentation/
   python smote_balancing.py          # Creates SMOTE balanced dataset
   ```
   
   **Option B - LLM Balancing:**
   ```bash
   cd scripts/data_augmentation/
   python setup_llm_generation.py     # Setup API key & generate reviews
   ```
   
   **Option C - Demo LLM (no cost):**
   ```bash
   cd scripts/data_augmentation/
   python llm_balancing_demo.py       # Demo without API calls
   ```

## Key Datasets

| Dataset | Size | Purpose | Location |
|---------|------|---------|-----------|
| Raw reviews | 1.3M | Original TripAdvisor data | `data/Madrid/reviews.pkl` |
| Spanish sentiment | 964K | Labeled Spanish reviews | `data/Madrid_spanish_reviews_sentiment.pkl` |
| 10K sample | 10K | Working dataset | `data/Madrid_stratified_sample_10000.pkl` |
| BERT embeddings | 2,683×768 | Feature vectors | `data/Madrid_neutral_negative_bert_embeddings.npy` |
| SMOTE balanced | 21,951 | SMOTE augmented | `data/Madrid_balanced_smote_dataset.pkl` |

## Data Balancing Comparison

| Method | Technique | Cost | Quality | Samples Generated |
|--------|-----------|------|---------|-------------------|
| **SMOTE** | Mathematical interpolation | Free | Synthetic vectors | 11,951 |
| **GPT-5 Mini** | LLM text generation | ~$3.67 | Authentic Spanish | 11,951 |

## Next Steps

1. **Implement ML models** in `scripts/model_training/`
2. **Compare SMOTE vs LLM** performance on classification tasks
3. **Evaluate** impact of different balancing methods
4. **Document results** for thesis research

## Technical Notes

- All scripts use **relative paths** (`../../data/`) from their subdirectories
- **BERT model**: Spanish Whole Word Masking (uncased)
- **LLM model**: GPT-5 Mini (latest OpenAI model, August 2025)
- **Class imbalance**: Extreme (6:1 ratio) - perfect for imbalance research
- **Language**: Spanish restaurant reviews from Madrid TripAdvisor