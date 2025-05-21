import json
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import wandb
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BengaliCodeDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the processed data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare the labels (same as input_ids for causal language modeling)
        labels = encodings.input_ids.clone()
        
        # Create attention mask
        attention_mask = encodings.attention_mask
        
        return {
            'input_ids': encodings.input_ids[0],
            'attention_mask': attention_mask[0],
            'labels': labels[0]
        }

class ModelTrainer:
    def __init__(self):
        self.data_dir = Path('data/raw')
        self.tokenizer_dir = Path('outputs/tokenizer')
        self.output_dir = Path('outputs/model')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        self.max_length = 2048
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-5
        self.num_train_epochs = 3
        self.warmup_steps = 100
        self.save_steps = 1000
        self.eval_steps = 500

    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project="bengali-code-llm",
            name="tinyllama-bengali-code",
            config={
                "model_name": self.model_name,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_train_epochs
            }
        )

    def prepare_model_and_tokenizer(self):
        """Load and prepare the model and tokenizer"""
        logger.info("Loading tokenizer and model")
        
        # Load the custom tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            model_max_length=self.max_length
        )
        
        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Resize token embeddings to match our tokenizer
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer

    def create_datasets(self, tokenizer):
        """Create training and validation datasets"""
        logger.info("Creating datasets")
        
        # Load the processed data
        data_path = self.data_dir / 'processed_data.json'
        
        # Split data into train and validation
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            
        np.random.seed(42)
        np.random.shuffle(all_data)
        
        split_idx = int(len(all_data) * 0.9)  # 90% train, 10% validation
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Save split data
        train_path = self.data_dir / 'train.json'
        val_path = self.data_dir / 'validation.json'
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # Create datasets
        train_dataset = BengaliCodeDataset(train_path, tokenizer, self.max_length)
        val_dataset = BengaliCodeDataset(val_path, tokenizer, self.max_length)
        
        return train_dataset, val_dataset

    def create_training_arguments(self):
        """Create training arguments for the Trainer"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            report_to="wandb",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        )

    def train(self):
        """Main method to train the model"""
        try:
            # Initialize wandb
            self.setup_wandb()
            
            # Prepare model and tokenizer
            model, tokenizer = self.prepare_model_and_tokenizer()
            
            # Create datasets
            train_dataset, val_dataset = self.create_datasets(tokenizer)
            
            # Create training arguments
            training_args = self.create_training_arguments()
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # We're doing causal language modeling
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            # Train the model
            logger.info("Starting model training")
            trainer.train()
            
            # Save the final model
            trainer.save_model(str(self.output_dir / 'final'))
            tokenizer.save_pretrained(str(self.output_dir / 'final'))
            
            # Close wandb
            wandb.finish()
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
        finally:
            # Ensure wandb is properly closed
            if wandb.run is not None:
                wandb.finish()

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
