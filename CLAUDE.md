# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project focused on restaurant reviews from Madrid. The project contains a comprehensive dataset of restaurant reviews, items, and users scraped from TripAdvisor, stored as pandas pickle files. The main analysis is conducted in a Jupyter notebook that performs extensive text analysis on review data.

## Data Architecture

### Core Data Files
- `Madrid/reviews.pkl` - Contains 1,347,959 restaurant reviews with columns: reviewId, userId, itemId, title, text, date, rating, language, images, url
- `Madrid/items.pkl` - Restaurant/item data (currently commented out in analysis)
- `Madrid/users.pkl` - User data (currently commented out in analysis)
- `Madrid.zip` - Compressed archive of the Madrid dataset

### Data Schema
The reviews dataset is the primary focus with these key characteristics:
- **Text column**: Main analysis target containing review text in multiple languages (primarily Spanish)
- **Rating system**: Uses values 10, 20, 30, 40, 50 (not 1-5 scale)
- **Language**: Multi-lingual with Spanish ('es') being dominant
- **Date range**: Spans multiple years with temporal analysis capabilities
- **No missing values**: Complete dataset with all fields populated

## Analysis Framework

### Jupyter Notebook Structure
The main notebook `Data_Extrction.ipynb` follows this analytical progression:
1. **Dataset Overview**: Basic statistics, memory usage, column analysis
2. **Text Analysis**: Length distributions, word counts, text categorization
3. **Language Analysis**: Distribution across languages, language-specific patterns
4. **Content Analysis**: Word frequency, Spanish stopword filtering, common terms
5. **Sentiment Analysis**: Custom positive/negative word lists, sentiment scoring by rating
6. **Pattern Analysis**: Regular expression matching for emails, URLs, special characters
7. **Data Quality**: Duplicate detection, suspicious pattern identification
8. **Visualizations**: Comprehensive charts and plots using matplotlib/seaborn
9. **Advanced Analysis**: Word clouds, temporal analysis, correlation studies

### Text Processing Approach
- **Language Focus**: Primary analysis on Spanish reviews (largest subset)
- **Cleaning Pipeline**: Lowercase conversion, URL/mention removal, whitespace normalization
- **Stopword Handling**: Custom Spanish stopword list for meaningful word extraction
- **Sentiment Analysis**: Rule-based approach using predefined positive/negative word lists

## Working with the Data

### Loading Data
```python
import pandas as pd
city = "Madrid"
reviews = pd.read_pickle(f"{city}/reviews.pkl")
items = pd.read_pickle(f"{city}/items.pkl")  # Optional
users = pd.read_pickle(f"{city}/users.pkl")  # Optional
```

### Required Libraries
The analysis requires these Python packages:
- pandas - Data manipulation and analysis
- matplotlib - Basic plotting
- seaborn - Statistical visualizations
- numpy - Numerical operations
- wordcloud - Word cloud generation (optional)
- collections.Counter - Word frequency analysis
- re - Regular expression pattern matching

### Memory Considerations
- The reviews dataset uses ~103 MB of memory
- Text analysis creates additional derived columns (text_length, word_count, etc.)
- For large-scale analysis, consider sampling for performance-intensive operations
- Visualization sampling is implemented for scatter plots (5000 sample limit)

## Analysis Patterns

### Text Length Categorization
The analysis uses these text length categories:
- Very Short (≤50 characters)
- Short (51-150 characters)  
- Medium (151-300 characters)
- Long (301-500 characters)
- Very Long (>500 characters)

### Rating Analysis Approach
- Ratings correlate with text length and sentiment indicators
- Higher ratings typically contain more positive vocabulary
- Text patterns (exclamation marks, caps) vary by rating level

### Quality Assessment Framework
The notebook includes comprehensive data quality checks:
- Empty or minimal text detection
- Duplicate text identification
- Suspicious pattern recognition (only punctuation, repeated characters, etc.)
- Language consistency validation

## Development Commands

### Running Jupyter Notebooks
Start Jupyter notebook server:
```bash
jupyter notebook
```

### Common Development Tasks
Execute individual notebook cells for targeted analysis:
- **Data_Extrction.ipynb**: Comprehensive review analysis and exploration
- **Sentiment_Analysis_Spanish_Reviews.ipynb**: Spanish sentiment analysis and class imbalance study

### Data Loading Pattern
Standard pattern for loading Madrid dataset:
```python
import pandas as pd
city = "Madrid"
reviews = pd.read_pickle(f"{city}/reviews.pkl")
# items = pd.read_pickle(f"{city}/items.pkl")  # Optional
# users = pd.read_pickle(f"{city}/users.pkl")   # Optional
```

## Development Notes

### Notebook Environment
- Designed for Jupyter notebook execution
- Uses IPython display utilities
- Handles missing dependencies gracefully (WordCloud fallback)
- Implements warning suppression for cleaner output

### Performance Considerations
- Large dataset (1.3M+ reviews, ~1GB memory usage)
- Strategic sampling for visualizations (5000 sample limit)
- Memory-efficient operations using pandas `.copy()` and filtering
- Text analysis on 964K Spanish reviews specifically

### Analysis Architecture
- **Two-stage analysis**: Basic exploration → Advanced sentiment analysis
- **Modular design**: Each notebook section can run independently
- **Language-specific processing**: Spanish text analysis with custom stopwords
- **Class imbalance focus**: Detailed imbalance metrics for thesis research

### Visualization Strategy
- Two-tier approach: basic analysis + advanced visualizations
- Comprehensive subplot layouts for overview dashboards
- Performance optimization through strategic sampling
- Fallback options when optional libraries unavailable

### Extensibility Considerations
- Modular analysis sections allow selective execution
- Language-agnostic framework can extend beyond Spanish
- Sentiment analysis framework can incorporate additional word lists
- Temporal analysis framework supports trend identification