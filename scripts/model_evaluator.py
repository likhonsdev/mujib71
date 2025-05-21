import json
from pathlib import Path
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.model_dir = Path('outputs/model/final')
        self.output_dir = Path('outputs/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test prompts for different scenarios
        self.test_prompts = [
            # Programming task prompts
            {
                "type": "code_generation",
                "prompt": "একটি পাইথন ফাংশন লিখুন যা একটি সংখ্যার ফ্যাক্টরিয়াল বের করে।",
                "expected": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)"""
            },
            {
                "type": "code_explanation",
                "prompt": "নিচের কোডটি ব্যাখ্যা করুন:\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
                "expected": "এই কোডটি বাবল সর্ট অ্যালগরিদম বাস্তবায়ন করে। এটি একটি অ্যারেকে ক্রমানুসারে সাজায়।"
            },
            {
                "type": "error_fix",
                "prompt": "এই কোডে ভুল আছে, ঠিক করুন:\ndef calculate_sum(numbers)\n    total = 0\n    for num in numbers\n        total += num\n    return total",
                "expected": """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total"""
            },
            # Algorithm explanation prompts
            {
                "type": "algorithm_explanation",
                "prompt": "বাইনারি সার্চ অ্যালগরিদম কীভাবে কাজ করে সেটি ব্যাখ্যা করুন।",
                "expected": "বাইনারি সার্চ একটি দক্ষ অ্যালগরিদম যা সর্টেড অ্যারেতে একটি এলিমেন্ট খোঁজে। এটি প্রতিবার অ্যারের মধ্যবর্তী এলিমেন্ট চেক করে এবং সার্চ স্পেস অর্ধেক করে কমিয়ে ফেলে।"
            }
        ]
        
        # Evaluation metrics
        self.bleu = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        logger.info("Loading model and tokenizer")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        return model, tokenizer

    def generate_response(self, model, tokenizer, prompt: str, max_length: int = 512) -> str:
        """Generate response for a given prompt"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate with better parameters for code generation
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""

    def calculate_metrics(self, generated: str, expected: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            # Calculate BLEU score
            bleu_score = self.bleu.corpus_score(
                [generated],
                [[expected]]
            ).score / 100.0
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(generated, expected)
            
            return {
                'bleu': bleu_score,
                'rouge1_f': rouge_scores['rouge1'].fmeasure,
                'rouge2_f': rouge_scores['rouge2'].fmeasure,
                'rougeL_f': rouge_scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'bleu': 0.0,
                'rouge1_f': 0.0,
                'rouge2_f': 0.0,
                'rougeL_f': 0.0
            }

    def evaluate(self):
        """Main method to evaluate the model"""
        try:
            # Initialize wandb for tracking
            wandb.init(project="bengali-code-llm", name="model-evaluation")
            
            # Load model and tokenizer
            model, tokenizer = self.load_model_and_tokenizer()
            
            # Store evaluation results
            results = []
            
            # Evaluate on test prompts
            for prompt_data in tqdm(self.test_prompts, desc="Evaluating prompts"):
                prompt_type = prompt_data["type"]
                prompt = prompt_data["prompt"]
                expected = prompt_data["expected"]
                
                # Generate response
                generated = self.generate_response(model, tokenizer, prompt)
                
                # Calculate metrics
                metrics = self.calculate_metrics(generated, expected)
                
                # Store result
                result = {
                    "type": prompt_type,
                    "prompt": prompt,
                    "generated": generated,
                    "expected": expected,
                    **metrics
                }
                results.append(result)
                
                # Log to wandb
                wandb.log({
                    f"{prompt_type}_bleu": metrics['bleu'],
                    f"{prompt_type}_rouge1": metrics['rouge1_f'],
                    f"{prompt_type}_rouge2": metrics['rouge2_f'],
                    f"{prompt_type}_rougeL": metrics['rougeL_f']
                })
            
            # Calculate average metrics by type
            df = pd.DataFrame(results)
            avg_metrics = df.groupby('type')[['bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f']].mean()
            
            # Save detailed results
            results_path = self.output_dir / 'evaluation_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Save average metrics
            metrics_path = self.output_dir / 'average_metrics.csv'
            avg_metrics.to_csv(metrics_path)
            
            # Log final averages to wandb
            wandb.log({
                "avg_bleu": df['bleu'].mean(),
                "avg_rouge1": df['rouge1_f'].mean(),
                "avg_rouge2": df['rouge2_f'].mean(),
                "avg_rougeL": df['rougeL_f'].mean()
            })
            
            # Close wandb
            wandb.finish()
            
            logger.info(f"Evaluation completed. Results saved to {self.output_dir}")
            
            # Return average metrics
            return avg_metrics.to_dict()
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
        finally:
            # Ensure wandb is properly closed
            if wandb.run is not None:
                wandb.finish()

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()
