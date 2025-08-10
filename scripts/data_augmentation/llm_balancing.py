import pandas as pd
import numpy as np
import openai
import json
import time
import random
import re
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LLM-BASED DATA BALANCING WITH GPT-4O MINI AND RAG")
print("="*70)

class ReviewRAG:
    """Retrieval-Augmented Generation system for restaurant reviews"""
    
    def __init__(self, reviews_df: pd.DataFrame, embeddings: np.ndarray):
        self.reviews_df = reviews_df
        self.embeddings = embeddings
        self.sentiment_reviews = {
            'negative': reviews_df[reviews_df['sentiment'] == 'negative'],
            'neutral': reviews_df[reviews_df['sentiment'] == 'neutral']
        }
        
    def get_similar_reviews(self, sentiment: str, n_samples: int = 5) -> List[Dict]:
        """Get similar reviews for a given sentiment"""
        sentiment_df = self.sentiment_reviews[sentiment]
        
        if len(sentiment_df) < n_samples:
            selected = sentiment_df
        else:
            selected = sentiment_df.sample(n=n_samples, random_state=random.randint(1, 1000))
        
        examples = []
        for _, review in selected.iterrows():
            examples.append({
                'text': review['text'][:300] + '...' if len(review['text']) > 300 else review['text'],
                'rating': review['rating'],
                'length': review['text_length']
            })
        
        return examples

class LLMReviewGenerator:
    """LLM-based review generator with GPT-5 Mini"""
    
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        else:
            print("⚠️  Warning: OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.model = "gpt-5-mini"  # Using GPT-5 mini - latest model from OpenAI
        self.request_count = 0
        self.total_tokens = 0
        
    def create_generation_prompt(self, sentiment: str, examples: List[Dict], target_length: str) -> str:
        """Create a detailed prompt for generating authentic Spanish reviews"""
        
        sentiment_descriptions = {
            'negative': {
                'spanish': 'negativa',
                'characteristics': 'mala comida, mal servicio, precios altos, experiencia decepcionante',
                'tone': 'frustrado, decepcionado, crítico',
                'rating_range': '10-20'
            },
            'neutral': {
                'spanish': 'neutral',
                'characteristics': 'experiencia promedio, algunos aspectos buenos y otros no tanto',
                'tone': 'equilibrado, objetivo, con comentarios mixtos',
                'rating_range': '30'
            }
        }
        
        length_specs = {
            'short': '50-150 caracteres (muy breve, directo)',
            'medium': '150-350 caracteres (detallado pero conciso)', 
            'long': '350-600 caracteres (muy detallado, experiencia completa)'
        }
        
        info = sentiment_descriptions[sentiment]
        
        examples_text = "\n".join([
            f"Ejemplo {i+1} (Rating {ex['rating']}, {ex['length']} chars): {ex['text']}"
            for i, ex in enumerate(examples)
        ])
        
        prompt = f"""Eres un experto en generar reseñas auténticas de restaurantes en español de España. 

TAREA: Genera UNA reseña {info['spanish']} completamente original y auténtica.

CARACTERÍSTICAS DE LA RESEÑA:
- Sentimiento: {info['spanish']} ({info['tone']})
- Rating esperado: {info['rating_range']}
- Longitud: {length_specs[target_length]}
- Temas típicos: {info['characteristics']}

EJEMPLOS REALES DE REFERENCIA:
{examples_text}

INSTRUCCIONES CRÍTICAS:
1. Escribe en español de España (usa "vosotros", "vale", "guay", etc.)
2. Menciona aspectos específicos: comida, servicio, ambiente, precios
3. Incluye detalles auténticos (nombres de platos, situaciones específicas)
4. Usa vocabulario variado y natural
5. NO copies frases exactas de los ejemplos
6. Haz que suene como una experiencia real personal
7. Incluye emociones y opiniones genuinas
8. Varía el estilo de escritura

RESPUESTA REQUERIDA:
Devuelve SOLO el texto de la reseña, sin explicaciones adicionales."""

        return prompt
    
    def generate_review(self, sentiment: str, examples: List[Dict], target_length: str, max_retries: int = 3) -> Dict[str, Any]:
        """Generate a single review using GPT-4o mini"""
        
        prompt = self.create_generation_prompt(sentiment, examples, target_length)
        
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Eres un experto generador de reseñas auténticas de restaurantes españoles."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.9,  # High creativity
                    top_p=0.95,
                    frequency_penalty=0.3,  # Reduce repetition
                    presence_penalty=0.3    # Encourage novelty
                )
                
                generated_text = response.choices[0].message.content.strip()
                
                # Clean the generated text
                generated_text = self.clean_generated_text(generated_text)
                
                self.request_count += 1
                self.total_tokens += response.usage.total_tokens
                
                # Create review metadata
                review_data = {
                    'text': generated_text,
                    'text_clean': self.clean_text_for_analysis(generated_text),
                    'sentiment': sentiment,
                    'rating': self.assign_rating(sentiment),
                    'text_length': len(generated_text),
                    'word_count': len(generated_text.split()),
                    'target_length': target_length,
                    'generated_by': 'GPT-5-mini',
                    'generation_attempt': attempt + 1
                }
                
                return review_data
                
            except Exception as e:
                print(f"  Error in attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
        
        return None
    
    def clean_generated_text(self, text: str) -> str:
        """Clean and format generated text"""
        # Remove any markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Remove any quotes around the text
        text = text.strip('"\'')
        
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text_for_analysis(self, text: str) -> str:
        """Clean text for analysis (similar to original preprocessing)"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def assign_rating(self, sentiment: str) -> int:
        """Assign rating based on sentiment"""
        if sentiment == 'negative':
            return random.choice([10, 20])
        elif sentiment == 'neutral':
            return 30
        else:
            return random.choice([40, 50])
    
    def generate_batch(self, sentiment: str, rag_system: ReviewRAG, count: int, batch_size: int = 10) -> List[Dict]:
        """Generate a batch of reviews with progress tracking"""
        
        generated_reviews = []
        length_options = ['short', 'medium', 'long']
        
        print(f"\nGenerating {count} {sentiment} reviews...")
        
        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            print(f"  Batch {i//batch_size + 1}: Generating {batch_count} reviews...")
            
            for j in range(batch_count):
                # Get fresh examples for each review
                examples = rag_system.get_similar_reviews(sentiment, n_samples=3)
                
                # Vary the target length
                target_length = random.choice(length_options)
                
                # Generate review
                review = self.generate_review(sentiment, examples, target_length)
                
                if review:
                    review['id'] = f"LLM_{sentiment}_{i+j+1}"
                    generated_reviews.append(review)
                    print(f"    ✓ Generated review {len(generated_reviews)}/{count}")
                else:
                    print(f"    ✗ Failed to generate review {i+j+1}")
                
                # Rate limiting
                time.sleep(1)  # 1 second between requests
            
            print(f"  Batch completed. Total: {len(generated_reviews)}/{count}")
            
            # Longer pause between batches
            if i + batch_size < count:
                print("  Pausing between batches...")
                time.sleep(5)
        
        return generated_reviews

def main():
    print("\n1. LOADING DATA AND SETTING UP RAG SYSTEM")
    print("-" * 50)
    
    # Load data
    neutral_negative_df = pd.read_pickle("../../data/Madrid_neutral_negative_sample.pkl")
    embeddings = np.load("../../data/Madrid_neutral_negative_bert_embeddings.npy")
    original_sample = pd.read_pickle("../../data/Madrid_stratified_sample_10000.pkl")
    
    print(f"Loaded {len(neutral_negative_df)} neutral/negative reviews with embeddings")
    print(f"Original sample distribution:")
    original_counts = original_sample['sentiment'].value_counts()
    print(original_counts)
    
    # Initialize RAG system
    rag_system = ReviewRAG(neutral_negative_df, embeddings)
    print("✓ RAG system initialized")
    
    # Initialize LLM generator
    llm_generator = LLMReviewGenerator()
    print("✓ LLM generator initialized")
    
    print("\n2. CALCULATING GENERATION TARGETS")
    print("-" * 50)
    
    # Calculate how many reviews to generate for balancing
    positive_count = original_counts['positive']
    current_negative = original_counts['negative'] 
    current_neutral = original_counts['neutral']
    
    negative_needed = positive_count - current_negative
    neutral_needed = positive_count - current_neutral
    
    print(f"Target count per class: {positive_count}")
    print(f"Negative reviews needed: {negative_needed}")
    print(f"Neutral reviews needed: {neutral_needed}")
    print(f"Total LLM reviews to generate: {negative_needed + neutral_needed}")
    
    # Ask user for confirmation due to cost implications
    print(f"\n⚠️  COST ESTIMATION (GPT-5 Mini):")
    print(f"Expected API calls: ~{negative_needed + neutral_needed}")
    
    # More accurate cost estimation
    avg_input_tokens_per_request = 800  # RAG context + prompt
    avg_output_tokens_per_request = 100  # Generated review
    
    total_input_tokens = (negative_needed + neutral_needed) * avg_input_tokens_per_request
    total_output_tokens = (negative_needed + neutral_needed) * avg_output_tokens_per_request
    
    # GPT-5 Mini pricing: $0.25/1M input, $2.00/1M output
    input_cost = (total_input_tokens / 1_000_000) * 0.25
    output_cost = (total_output_tokens / 1_000_000) * 2.00
    estimated_cost = input_cost + output_cost
    
    print(f"Input tokens: ~{total_input_tokens:,} (${input_cost:.2f})")
    print(f"Output tokens: ~{total_output_tokens:,} (${output_cost:.2f})")
    print(f"Total estimated cost: ~${estimated_cost:.2f} USD")
    
    response = input("\nProceed with LLM generation? (y/N): ").lower().strip()
    if response != 'y':
        print("Generation cancelled by user.")
        return
    
    print("\n3. GENERATING LLM REVIEWS")
    print("-" * 50)
    
    all_generated_reviews = []
    
    # Generate negative reviews
    if negative_needed > 0:
        negative_reviews = llm_generator.generate_batch(
            'negative', rag_system, negative_needed, batch_size=5
        )
        all_generated_reviews.extend(negative_reviews)
        print(f"✓ Generated {len(negative_reviews)} negative reviews")
    
    # Generate neutral reviews
    if neutral_needed > 0:
        neutral_reviews = llm_generator.generate_batch(
            'neutral', rag_system, neutral_needed, batch_size=5
        )
        all_generated_reviews.extend(neutral_reviews)
        print(f"✓ Generated {len(neutral_reviews)} neutral reviews")
    
    print(f"\n✓ Total LLM reviews generated: {len(all_generated_reviews)}")
    print(f"✓ API requests made: {llm_generator.request_count}")
    print(f"✓ Total tokens used: {llm_generator.total_tokens}")
    
    print("\n4. CREATING BALANCED DATASET")
    print("-" * 50)
    
    # Convert generated reviews to DataFrame
    llm_df = pd.DataFrame(all_generated_reviews)
    
    # Add missing columns to match original format
    llm_df['reviewId'] = llm_df['id']
    llm_df['date'] = '2024-LLM'
    llm_df['synthetic'] = True
    llm_df['images'] = '[]'
    llm_df['url'] = 'generated_by_llm'
    llm_df['userId'] = 'LLM_USER'
    llm_df['itemId'] = 99999999
    llm_df['title'] = llm_df['text'].str[:50] + '...'
    
    # Combine with original data
    original_positive = original_sample[original_sample['sentiment'] == 'positive'].copy()
    original_negative_neutral = neutral_negative_df.copy()
    
    # Add synthetic flag to original data
    original_positive['synthetic'] = False
    original_negative_neutral['synthetic'] = False
    
    # Combine all data
    balanced_llm_df = pd.concat([
        original_positive,
        original_negative_neutral,
        llm_df
    ], ignore_index=True)
    
    print(f"Balanced dataset created:")
    print(f"  Total samples: {len(balanced_llm_df)}")
    final_counts = balanced_llm_df['sentiment'].value_counts()
    print(final_counts)
    
    print("\n5. SAVING RESULTS")
    print("-" * 50)
    
    # Save balanced dataset
    llm_balanced_file = "Madrid_balanced_llm_dataset.pkl"
    balanced_llm_df.to_pickle(llm_balanced_file)
    print(f"✓ Balanced dataset saved: {llm_balanced_file}")
    
    llm_balanced_csv = "Madrid_balanced_llm_dataset.csv"
    balanced_llm_df.to_csv(llm_balanced_csv, index=False)
    print(f"✓ Balanced dataset CSV: {llm_balanced_csv}")
    
    # Save LLM-generated reviews separately
    llm_only_file = "Madrid_llm_generated_reviews.pkl"
    llm_df.to_pickle(llm_only_file)
    print(f"✓ LLM-generated reviews: {llm_only_file}")
    
    # Save metadata
    metadata = {
        'method': 'LLM_GPT5_mini_with_RAG',
        'model_used': 'gpt-5-mini',
        'original_distribution': {k: int(v) for k, v in original_counts.items()},
        'balanced_distribution': {k: int(v) for k, v in final_counts.items()},
        'llm_samples_generated': len(all_generated_reviews),
        'api_requests_made': llm_generator.request_count,
        'total_tokens_used': llm_generator.total_tokens,
        'generation_parameters': {
            'temperature': 0.9,
            'top_p': 0.95,
            'max_tokens': 500,
            'frequency_penalty': 0.3,
            'presence_penalty': 0.3
        },
        'rag_parameters': {
            'examples_per_generation': 3,
            'length_variations': ['short', 'medium', 'long']
        },
        'total_balanced_samples': len(balanced_llm_df),
        'files_created': [llm_balanced_file, llm_balanced_csv, llm_only_file]
    }
    
    metadata_file = "Madrid_llm_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata saved: {metadata_file}")
    
    print("\n6. CREATING ANALYSIS")
    print("-" * 50)
    
    # Analyze generated reviews
    print("Generated reviews analysis:")
    print(f"  Average text length: {llm_df['text_length'].mean():.1f} chars")
    print(f"  Average word count: {llm_df['word_count'].mean():.1f} words")
    print(f"  Length distribution by target:")
    for target_length in ['short', 'medium', 'long']:
        subset = llm_df[llm_df['target_length'] == target_length]
        if len(subset) > 0:
            print(f"    {target_length}: {subset['text_length'].mean():.1f} ± {subset['text_length'].std():.1f} chars")
    
    # Sample generated reviews
    print(f"\nSample generated reviews:")
    for sentiment in ['negative', 'neutral']:
        sentiment_samples = llm_df[llm_df['sentiment'] == sentiment].sample(min(2, len(llm_df[llm_df['sentiment'] == sentiment])))
        print(f"\n{sentiment.upper()} examples:")
        for _, review in sentiment_samples.iterrows():
            print(f"  • ({review['text_length']} chars): {review['text'][:150]}...")
    
    print("\n" + "="*70)
    print("LLM-BASED BALANCING COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()