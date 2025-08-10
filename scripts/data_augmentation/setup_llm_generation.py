import os
import getpass

print("="*60)
print("LLM GENERATION SETUP")
print("="*60)

print("""
This script will generate authentic Spanish restaurant reviews using GPT-5 Mini
to balance your dataset. The process includes:

1. RAG-based context retrieval from existing reviews
2. Authentic Spanish review generation with varied lengths
3. Batch processing to handle API limits
4. Cost-efficient generation with GPT-5 Mini (latest OpenAI model)

REQUIREMENTS:
- OpenAI API key
- Sufficient API credits (~$3-4 estimated cost with GPT-5 Mini)
- Internet connection

The script will generate:
- ~5,842 negative reviews 
- ~6,109 neutral reviews
- Total: ~11,951 synthetic reviews
""")

def setup_api_key():
    """Setup OpenAI API key"""
    
    # Check if already set in environment
    if 'OPENAI_API_KEY' in os.environ:
        print("✓ OpenAI API key found in environment variables")
        return True
    
    print("\nOpenAI API Key Setup:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Copy and paste it below")
    print("4. Make sure your account has sufficient credits")
    
    while True:
        api_key = getpass.getpass("\nEnter your OpenAI API key (hidden input): ").strip()
        
        if not api_key:
            print("❌ Empty API key. Please try again.")
            continue
            
        if not api_key.startswith('sk-'):
            print("❌ API key should start with 'sk-'. Please check and try again.")
            continue
            
        if len(api_key) < 40:
            print("❌ API key seems too short. Please check and try again.")
            continue
        
        # Set the environment variable for the current session
        os.environ['OPENAI_API_KEY'] = api_key
        print("✓ API key set successfully")
        
        return True

def run_generation():
    """Run the LLM generation process"""
    print("\n" + "="*60)
    print("STARTING LLM GENERATION PROCESS")
    print("="*60)
    
    # Import and run the main LLM balancing script
    try:
        exec(open('llm_balancing.py').read())
    except FileNotFoundError:
        print("❌ llm_balancing.py not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Error running LLM generation: {str(e)}")
        return False
    
    return True

def main():
    print("Setting up LLM generation environment...")
    
    # Setup API key
    if not setup_api_key():
        print("❌ Failed to setup API key. Exiting.")
        return
    
    # Confirm before proceeding
    print(f"\n⚠️  FINAL CONFIRMATION:")
    print(f"This will make ~12,000 API calls to OpenAI GPT-5 Mini")
    print(f"Estimated cost: ~$3-4 USD (with GPT-5 Mini pricing)")
    print(f"  - Input tokens: ~$2.39 USD")
    print(f"  - Output tokens: ~$1.26 USD") 
    print(f"Estimated time: 2-4 hours (with rate limiting)")
    
    confirm = input(f"\nProceed with generation? (type 'yes' to continue): ").lower().strip()
    
    if confirm != 'yes':
        print("❌ Generation cancelled. Run this script again when ready.")
        return
    
    # Run the generation
    success = run_generation()
    
    if success:
        print("\n✅ LLM generation completed successfully!")
        print("\nFiles created:")
        print("- Madrid_balanced_llm_dataset.pkl")
        print("- Madrid_balanced_llm_dataset.csv") 
        print("- Madrid_llm_generated_reviews.pkl")
        print("- Madrid_llm_metadata.json")
    else:
        print("\n❌ LLM generation failed. Check the error messages above.")

if __name__ == "__main__":
    main()