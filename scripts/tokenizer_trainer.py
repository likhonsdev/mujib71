import json
from pathlib import Path
import sentencepiece as spm
import logging
from typing import List, Dict
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenizerTrainer:
    def __init__(self):
        self.data_dir = Path('data/raw')
        self.output_dir = Path('outputs/tokenizer')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tokenizer configuration
        self.vocab_size = 32000
        self.character_coverage = 0.9999
        self.model_type = "unigram"
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "<s>", "</s>", "<pad>", "<unk>", "<mask>",
            "২০", "১০", "৫০", "১৫", "২৫",  # Common Bengali numbers
            "def", "class", "return", "if", "else", "for", "while",  # Code keywords
            "print", "input", "import", "from", "try", "except",
            "#", "//", "/*", "*/", "'''", '"""'  # Code comments
        ]

    def prepare_training_data(self) -> str:
        """Prepare text data for tokenizer training"""
        logger.info("Preparing training data for tokenizer")
        
        # Load processed data
        try:
            with open(self.data_dir / 'processed_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error("Processed data file not found. Run data collection first.")
            raise
            
        # Create temporary file for training
        train_file = self.output_dir / 'train.txt'
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in data:
                text = item['text']
                # Write one sentence per line
                sentences = text.split('।')  # Split on Bengali full stop
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:  # Skip empty sentences
                        f.write(sentence + '\n')
                        
        logger.info("Training data prepared successfully")
        return str(train_file)

    def train_tokenizer(self, train_file: str):
        """Train the SentencePiece tokenizer"""
        logger.info("Starting tokenizer training")
        
        # Prepare model prefix
        model_prefix = self.output_dir / "bengali_code"
        
        # Create training parameters
        params = {
            "--input": train_file,
            "--model_prefix": str(model_prefix),
            "--vocab_size": str(self.vocab_size),
            "--character_coverage": str(self.character_coverage),
            "--model_type": self.model_type,
            "--pad_id": 0,
            "--unk_id": 1,
            "--bos_id": 2,
            "--eos_id": 3,
            "--user_defined_symbols": ",".join(self.special_tokens),
            "--max_sentence_length": "4192",
            "--input_sentence_size": "5000000",
            "--shuffle_input_sentence": "true",
            "--normalization_rule_name": "identity"  # Preserve original text
        }
        
        # Convert parameters to command-line arguments
        args = []
        for key, value in params.items():
            args.append(key)
            args.append(value)
            
        try:
            # Train the tokenizer
            spm.SentencePieceTrainer.train(" ".join(args))
            logger.info("Tokenizer training completed successfully")
            
            # Create config files for HuggingFace compatibility
            self.create_huggingface_files(model_prefix)
            
        except Exception as e:
            logger.error(f"Failed to train tokenizer: {str(e)}")
            raise

    def create_huggingface_files(self, model_prefix: Path):
        """Create additional files needed for HuggingFace compatibility"""
        logger.info("Creating HuggingFace compatibility files")
        
        # Create tokenizer config
        tokenizer_config = {
            "model_max_length": 2048,
            "padding_side": "right",
            "truncation_side": "right",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "model_type": self.model_type,
            "vocab_size": self.vocab_size
        }
        
        with open(self.output_dir / "tokenizer_config.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
            
        # Create special tokens map
        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        }
        
        with open(self.output_dir / "special_tokens_map.json", 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
            
        logger.info("HuggingFace compatibility files created successfully")

    def train(self):
        """Main method to train the tokenizer"""
        try:
            # Prepare training data
            train_file = self.prepare_training_data()
            
            # Train tokenizer
            self.train_tokenizer(train_file)
            
            # Clean up temporary files
            if Path(train_file).exists():
                Path(train_file).unlink()
                
            logger.info("Tokenizer training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Tokenizer training pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = TokenizerTrainer()
    trainer.train()
