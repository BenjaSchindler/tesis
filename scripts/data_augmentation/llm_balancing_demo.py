import pandas as pd
import numpy as np
import json
import time
import random
import re
from typing import List, Dict, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LLM-BASED DATA BALANCING DEMO (WITHOUT API CALLS)")
print("="*70)

class DemoReviewGenerator:
    """Demo version that simulates LLM generation without API calls"""
    
    def __init__(self):
        self.request_count = 0
        self.total_tokens = 0
        
        # Pre-generated example reviews for demo
        self.demo_negative_reviews = [
            "El servicio fue lento y la comida llegó fría. Los precios son demasiado altos para la calidad que ofrecen. No recomiendo este lugar.",
            "Experiencia muy decepcionante. El camarero era maleducado y la pasta estaba sosa. Definitivamente no volveremos.",
            "La peor cena que he tenido en mucho tiempo. El pescado olía mal y el ambiente era desagradable. Eviten este restaurante.",
            "Reservamos mesa y nos hicieron esperar 45 minutos. Cuando llegó la comida, estaba quemada. El precio no justifica la calidad.",
            "Comida horrible, servicio pésimo. El bistec estaba duro como una suela. No entiendo las buenas reseñas que tiene."
        ]
        
        self.demo_neutral_reviews = [
            "Restaurante correcto, sin más. La comida estaba bien pero nada especial. El servicio fue normal, ni bueno ni malo.",
            "Experiencia regular. Algunos platos estaban ricos pero otros no tanto. Los precios son razonables para la zona.",
            "No está mal pero tampoco es para volver corriendo. La decoración es bonita pero la comida es del montón.",
            "Lugar decente para una comida rápida. El personal es amable aunque un poco lento. La relación calidad-precio es aceptable.",
            "Ambiente agradable y ubicación buena. La comida está bien aunque podría tener más sabor. En general, una experiencia normal."
        ]
    
    def generate_review_demo(self, sentiment: str, target_length: str) -> Dict[str, Any]:
        """Generate a demo review by selecting from pre-made examples"""
        
        if sentiment == 'negative':
            base_text = random.choice(self.demo_negative_reviews)
        else:
            base_text = random.choice(self.demo_neutral_reviews)
        
        # Simulate slight variations
        variations = [
            base_text,
            base_text + " El local necesita mejorar varios aspectos.",
            base_text.replace(".", ", aunque el ambiente era agradable."),
            base_text + " Quizás en otra ocasión sea diferente."
        ]
        
        final_text = random.choice(variations)
        
        # Adjust length based on target
        if target_length == 'short':
            final_text = final_text[:100] + "." if len(final_text) > 100 else final_text
        elif target_length == 'long':
            final_text += " La experiencia en general fue lo que esperábamos según las reseñas previas."
        
        self.request_count += 1
        self.total_tokens += len(final_text.split()) * 2  # Simulate token usage
        
        return {
            'text': final_text,
            'text_clean': final_text.lower().strip(),
            'sentiment': sentiment,
            'rating': 10 if sentiment == 'negative' else 30,
            'text_length': len(final_text),
            'word_count': len(final_text.split()),
            'target_length': target_length,
            'generated_by': 'Demo_Generator',
            'generation_attempt': 1
        }

def main():
    print("\n🎭 DEMO MODE: Simulating LLM generation without API calls")
    print("This demo shows how the LLM balancing process would work")
    print("-" * 70)
    
    print("\n1. LOADING DATA")
    print("-" * 40)
    
    # Load data
    try:
        original_sample = pd.read_pickle("../../data/Madrid_stratified_sample_10000.pkl")
        print(f"✓ Loaded original sample: {len(original_sample)} reviews")
        
        original_counts = original_sample['sentiment'].value_counts()
        print("Original distribution:")
        print(original_counts)
        
    except FileNotFoundError:
        print("❌ ../../data/Madrid_stratified_sample_10000.pkl not found")
        print("Please run the previous scripts to create the sample first")
        return
    
    print("\n2. DEMO GENERATION (SMALL SCALE)")
    print("-" * 40)
    
    # Generate just a few samples for demo
    demo_generator = DemoReviewGenerator()
    
    # Generate small demo samples
    demo_negative_count = 10
    demo_neutral_count = 8
    
    print(f"Generating {demo_negative_count} negative and {demo_neutral_count} neutral reviews for demo...")
    
    all_generated_reviews = []
    length_options = ['short', 'medium', 'long']
    
    # Generate negative reviews
    for i in range(demo_negative_count):
        target_length = random.choice(length_options)
        review = demo_generator.generate_review_demo('negative', target_length)
        review['id'] = f"DEMO_negative_{i+1}"
        all_generated_reviews.append(review)
        time.sleep(0.1)  # Simulate API delay
        print(f"  ✓ Generated negative review {i+1}/{demo_negative_count}")
    
    # Generate neutral reviews
    for i in range(demo_neutral_count):
        target_length = random.choice(length_options)
        review = demo_generator.generate_review_demo('neutral', target_length)
        review['id'] = f"DEMO_neutral_{i+1}"
        all_generated_reviews.append(review)
        time.sleep(0.1)  # Simulate API delay
        print(f"  ✓ Generated neutral review {i+1}/{demo_neutral_count}")
    
    print(f"\n✓ Demo generation completed!")
    print(f"✓ 'API requests' made: {demo_generator.request_count}")
    print(f"✓ 'Tokens' used: {demo_generator.total_tokens}")
    
    print("\n3. DEMO RESULTS ANALYSIS")
    print("-" * 40)
    
    # Convert to DataFrame
    demo_df = pd.DataFrame(all_generated_reviews)
    
    print("Generated reviews analysis:")
    print(f"  Total generated: {len(demo_df)} reviews")
    print(f"  Average text length: {demo_df['text_length'].mean():.1f} chars")
    print(f"  Average word count: {demo_df['word_count'].mean():.1f} words")
    
    print("\nLength distribution by target:")
    for target_length in ['short', 'medium', 'long']:
        subset = demo_df[demo_df['target_length'] == target_length]
        if len(subset) > 0:
            print(f"  {target_length}: {len(subset)} reviews, {subset['text_length'].mean():.1f} ± {subset['text_length'].std():.1f} chars")
    
    print("\nSample generated reviews:")
    for sentiment in ['negative', 'neutral']:
        sentiment_samples = demo_df[demo_df['sentiment'] == sentiment].head(2)
        print(f"\n{sentiment.upper()} examples:")
        for _, review in sentiment_samples.iterrows():
            print(f"  • [{review['text_length']} chars, {review['target_length']}]: {review['text']}")
    
    print("\n4. FULL SCALE PROJECTION")
    print("-" * 40)
    
    positive_count = original_counts['positive']
    current_negative = original_counts['negative'] 
    current_neutral = original_counts['neutral']
    
    negative_needed = positive_count - current_negative
    neutral_needed = positive_count - current_neutral
    total_needed = negative_needed + neutral_needed
    
    print(f"For full balancing, you would need to generate:")
    print(f"  Negative reviews: {negative_needed:,}")
    print(f"  Neutral reviews: {neutral_needed:,}")
    print(f"  Total LLM reviews: {total_needed:,}")
    
    # Cost estimation with GPT-5 Mini pricing
    avg_tokens_per_review = demo_df['word_count'].mean() * 2  # Rough estimate
    
    # Estimate input and output tokens separately
    avg_input_tokens_per_request = 800  # RAG context + prompt
    avg_output_tokens_per_request = avg_tokens_per_review
    
    total_input_tokens = total_needed * avg_input_tokens_per_request
    total_output_tokens = total_needed * avg_output_tokens_per_request
    
    # GPT-5 Mini pricing: $0.25/1M input, $2.00/1M output
    input_cost = (total_input_tokens / 1_000_000) * 0.25
    output_cost = (total_output_tokens / 1_000_000) * 2.00
    estimated_cost = input_cost + output_cost
    
    print(f"\nEstimated full generation:")
    print(f"  API requests: ~{total_needed:,}")
    print(f"  Input tokens: ~{total_input_tokens:,.0f}")
    print(f"  Output tokens: ~{total_output_tokens:,.0f}")
    print(f"  Total tokens: ~{total_input_tokens + total_output_tokens:,.0f}")
    print(f"  Input cost: ~${input_cost:.2f} USD")
    print(f"  Output cost: ~${output_cost:.2f} USD")
    print(f"  Total estimated cost: ~${estimated_cost:.2f} USD")
    print(f"  Estimated time: ~{total_needed/100:.1f} hours (with rate limiting)")
    
    print("\n5. NEXT STEPS")
    print("-" * 40)
    print("To run the full LLM generation:")
    print("1. Get an OpenAI API key from https://platform.openai.com/api-keys")
    print("2. Add sufficient credits to your OpenAI account")
    print("3. Run: python setup_llm_generation.py")
    print("4. Follow the prompts to enter your API key")
    print("5. Confirm the generation when prompted")
    
    # Save demo results
    demo_file = "Madrid_llm_generation_demo.pkl"
    demo_df.to_pickle(demo_file)
    print(f"\n✓ Demo results saved to: {demo_file}")
    
    demo_metadata = {
        'demo_mode': True,
        'samples_generated': len(demo_df),
        'full_scale_projection': {
            'negative_needed': int(negative_needed),
            'neutral_needed': int(neutral_needed),
            'total_needed': int(total_needed),
            'estimated_input_tokens': int(total_input_tokens),
            'estimated_output_tokens': int(total_output_tokens),
            'estimated_total_tokens': int(total_input_tokens + total_output_tokens),
            'estimated_cost_usd': round(estimated_cost, 2),
            'estimated_input_cost_usd': round(input_cost, 2),
            'estimated_output_cost_usd': round(output_cost, 2)
        }
    }
    
    with open("Madrid_llm_demo_metadata.json", 'w') as f:
        json.dump(demo_metadata, f, indent=2)
    
    print("✓ Demo metadata saved to: Madrid_llm_demo_metadata.json")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()