# Thesis Project: Spanish Restaurant Review Sentiment Analysis with Data Balancing

## 🎯 Project Overview

This thesis project focuses on addressing severe class imbalance in Spanish restaurant review sentiment analysis. Using a comprehensive dataset of 1.3M+ TripAdvisor reviews from Madrid, we implement and compare two advanced data balancing techniques: SMOTE (Synthetic Minority Oversampling Technique) and LLM-based generation using GPT-5 Mini.

### Key Problem
- **Severe Class Imbalance**: 73.2% positive, 14.8% negative, 12.1% neutral reviews
- **Imbalance Ratio**: 6.05:1 (extreme classification challenge)
- **Language**: Spanish restaurant reviews requiring specialized NLP techniques

## 🏗️ Project Architecture

```
📁 Tesis/
├── 📄 CLAUDE.md                    # AI assistant project instructions
├── 📄 README_Project_Structure.md   # Detailed technical documentation
├── 📄 README.md                     # This file
│
├── 📁 data/                         # Complete dataset pipeline
│   ├── Madrid.zip                   # Original 1.3M TripAdvisor reviews
│   ├── Madrid/                      # Raw datasets
│   │   ├── reviews.pkl             # 1,347,959 restaurant reviews
│   │   ├── items.pkl               # Restaurant metadata
│   │   └── users.pkl               # User data
│   │
│   ├── # Processed Datasets
│   ├── Madrid_spanish_reviews_sentiment.pkl     # 964K Spanish reviews with sentiment
│   ├── Madrid_stratified_sample_10000.pkl       # Working dataset (10K samples)
│   │
│   ├── # BERT Embeddings
│   ├── Madrid_neutral_negative_bert_embeddings.npy  # 768-dim Spanish BERT vectors
│   ├── Madrid_bert_embeddings_metadata.json         # Embedding specifications
│   │
│   ├── # SMOTE Balanced Data
│   ├── Madrid_balanced_smote_dataset.pkl        # 21,951 SMOTE-balanced samples
│   ├── Madrid_smote_embeddings.npy              # SMOTE-generated embeddings
│   ├── Madrid_SMOTE_analysis.png                # SMOTE visualization results
│   │
│   └── # LLM Generated Data
│       ├── Madrid_balanced_llm_dataset.pkl      # 21,951 GPT-5 Mini balanced samples
│       └── Madrid_llm_metadata.json             # LLM generation parameters
│
└── 📁 scripts/                      # Organized by data science workflow
    ├── 📁 data_exploration/         # Analysis and visualization
    │   ├── Data_Extrction.ipynb     # Comprehensive EDA (1.3M reviews)
    │   └── Sentiment_Analysis_Spanish_Reviews.ipynb  # Class imbalance analysis
    │
    ├── 📁 data_extraction/          # Preprocessing and feature extraction
    │   ├── sample_and_embeddings.py # Stratified sampling (964K → 10K)
    │   └── bert_embeddings.py       # Spanish BERT embedding generation
    │
    ├── 📁 data_augmentation/        # Data balancing implementations
    │   ├── smote_balancing.py       # SMOTE oversampling technique
    │   ├── llm_balancing.py         # GPT-5 Mini authentic text generation
    │   ├── llm_balancing_demo.py    # Cost-free LLM demonstration
    │   └── setup_llm_generation.py  # OpenAI API setup and cost estimation
    │
    └── 📁 model_training/           # Machine learning implementation
        └── (Ready for ML model comparison scripts)
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python environment
python 3.8+

# Core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
pip install transformers torch tqdm
pip install imbalanced-learn  # For SMOTE
pip install openai           # For LLM generation
pip install jupyter          # For notebooks

# Optional: For advanced visualizations
pip install wordcloud
```

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/BenjaSchindler/tesis.git
cd tesis
```

2. **Run the complete data pipeline**:

```bash
# Step 1: Create working dataset
cd scripts/data_extraction/
python sample_and_embeddings.py    # Creates 10K stratified sample
python bert_embeddings.py          # Generates Spanish BERT embeddings

# Step 2A: SMOTE balancing (computational approach)
cd ../data_augmentation/
python smote_balancing.py          # Creates 21,951 balanced samples via interpolation

# Step 2B: LLM balancing (authentic text generation)
cd ../data_augmentation/
python setup_llm_generation.py     # GPT-5 Mini generation (~$3.67 cost)

# Step 3: Model training (implementation phase)
cd ../model_training/
# [Ready for ML model comparison scripts]
```

3. **Explore the data**:
```bash
# Data analysis and visualization
jupyter notebook scripts/data_exploration/
```

## 🔬 Research Methodology

### Data Balancing Techniques Comparison

| Method | Technique | Cost | Quality | Samples Generated |
|--------|-----------|------|---------|-------------------|
| **SMOTE** | Mathematical interpolation in 768D BERT space | Free | Synthetic vectors | 11,951 samples |
| **GPT-5 Mini** | LLM text generation with RAG system | ~$3.67 | Authentic Spanish text | 11,951 samples |

### Technical Specifications

- **BERT Model**: `dccuchile/bert-base-spanish-wwm-uncased` (Spanish Whole Word Masking)
- **Embedding Dimension**: 768 (transformer-based representations)
- **SMOTE Parameters**: k=5 neighbors, random_state=42
- **LLM Model**: GPT-5 Mini (latest OpenAI model, August 2025)
- **LLM Pricing**: $0.25/1M input tokens, $2.00/1M output tokens

## 📊 Dataset Characteristics

### Original Distribution (964K Spanish Reviews)
- **Positive**: 706,214 (73.2%) - Ratings 40-50
- **Negative**: 142,651 (14.8%) - Ratings 10-20  
- **Neutral**: 116,493 (12.1%) - Rating 30
- **Language**: Spanish (TripAdvisor Madrid restaurants)
- **Time Period**: Multi-year review collection

### Class Imbalance Challenge
- **Imbalance Ratio**: 6.05:1 (severe)
- **Classification Difficulty**: Extreme minority class underrepresentation
- **Business Impact**: Critical for negative review detection in hospitality

## 🔧 Usage Examples

### SMOTE Balancing
```python
# Load and apply SMOTE to BERT embeddings
from imblearn.over_sampling import SMOTE
import numpy as np

embeddings = np.load("data/Madrid_neutral_negative_bert_embeddings.npy")
smote = SMOTE(sampling_strategy={'negative': 7317, 'neutral': 7317}, 
              k_neighbors=5, random_state=42)
X_balanced, y_balanced = smote.fit_resample(embeddings, labels)
```

### LLM Generation
```python
# Generate authentic Spanish reviews with GPT-5 Mini
from scripts.data_augmentation.llm_balancing import LLMReviewGenerator

generator = LLMReviewGenerator(api_key="your-openai-key")
review = generator.generate_review(
    sentiment='negative',
    examples=rag_examples,
    target_length='medium'
)
```

## 📈 Expected Results

### Research Hypotheses
1. **SMOTE Performance**: Effective for traditional ML algorithms but may lack semantic diversity
2. **LLM Performance**: Higher quality text generation but requires API costs and careful prompt engineering
3. **Hybrid Approach**: Combination of both methods may yield optimal results

### Evaluation Metrics
- **Classification Performance**: Precision, Recall, F1-Score per class
- **Imbalance Handling**: Balanced Accuracy, AUC-ROC for minority classes
- **Text Quality**: Semantic similarity, linguistic authenticity (Spanish)
- **Computational Efficiency**: Training time, inference speed

## 🛠️ Development Commands

```bash
# Data pipeline execution
make sample          # Create stratified sample
make embeddings      # Generate BERT embeddings  
make smote          # Apply SMOTE balancing
make llm            # Generate LLM reviews (requires API key)

# Analysis and visualization
make explore        # Launch Jupyter notebooks
make visualize      # Generate analysis plots

# Model training (implementation phase)
make train          # Train comparison models
make evaluate       # Performance evaluation
```

## 📚 Technical Documentation

- **[README_Project_Structure.md](README_Project_Structure.md)**: Complete technical workflow documentation
- **[CLAUDE.md](CLAUDE.md)**: AI assistant development guidelines
- **Jupyter Notebooks**: Comprehensive EDA and analysis in `scripts/data_exploration/`

## 🤝 Contributing

This thesis project follows academic research standards:

1. **Data Ethics**: TripAdvisor public review data with privacy considerations
2. **Reproducibility**: Fixed random seeds, documented parameters
3. **Version Control**: Git-based development with comprehensive documentation
4. **Code Quality**: PEP 8 standards, comprehensive commenting

## 📄 License

Academic thesis project - Universidad [Name] - 2024

## 🔗 Related Work

- **SMOTE**: Chawla et al. (2002) - Synthetic Minority Oversampling Technique
- **Spanish BERT**: University of Chile BERT for Spanish language processing
- **LLM Data Augmentation**: Recent advances in large language model text generation
- **Sentiment Analysis**: Spanish restaurant review sentiment classification

## 📧 Contact

**Author**: Benjamin Schindler  
**Institution**: [University Name]  
**Thesis Supervisor**: [Supervisor Name]  
**GitHub**: [@BenjaSchindler](https://github.com/BenjaSchindler)

---

*This project demonstrates advanced techniques for addressing severe class imbalance in Spanish NLP tasks, contributing to both academic research and practical sentiment analysis applications in the hospitality industry.*
