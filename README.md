# Bengali-Code LLM Training Pipeline

A comprehensive pipeline for training a Bengali language model specialized in code understanding and generation. The model is fine-tuned on Bengali programming tutorials, documentation, and code examples.

## 🌟 Features

- Automated data collection from Bengali Wikipedia and Prothom Alo
- Custom tokenizer training with SentencePiece for Bengali text and code
- Model fine-tuning using TinyLlama base model
- Comprehensive evaluation suite for Bengali code generation
- GitHub Actions workflow for automated training
- Weights & Biases integration for experiment tracking

## 📋 Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- Internet connection for data collection

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bengali-code-llm.git
cd bengali-code-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
export WANDB_API_KEY="your_wandb_key_here"
```

4. Run the complete pipeline:
```bash
# Collect data
python scripts/data_collector.py

# Train tokenizer
python scripts/tokenizer_trainer.py

# Train model
python scripts/model_trainer.py

# Evaluate model
python scripts/model_evaluator.py
```

## 🏗️ Pipeline Components

### Data Collection (`scripts/data_collector.py`)
- Scrapes Bengali text from Wikipedia and Prothom Alo
- Implements rate limiting and error handling
- Outputs processed data in JSON format

### Tokenizer Training (`scripts/tokenizer_trainer.py`)
- Uses SentencePiece for tokenizer training
- Custom vocabulary with Bengali and code tokens
- Generates HuggingFace-compatible tokenizer files

### Model Training (`scripts/model_trainer.py`)
- Fine-tunes TinyLlama model
- Implements efficient training with gradient accumulation
- Supports mixed precision training
- Integrates with Weights & Biases for tracking

### Model Evaluation (`scripts/model_evaluator.py`)
- Comprehensive evaluation suite
- Tests code generation capabilities
- Measures BLEU and ROUGE scores
- Generates detailed evaluation reports

## 📊 Training Metrics

The training progress can be monitored through Weights & Biases:
- Loss curves
- Evaluation metrics
- Generated samples
- Resource utilization

## 🔄 GitHub Actions Workflow

The repository includes an automated training pipeline that:
- Runs daily to incorporate new data
- Executes the complete training pipeline
- Uploads model artifacts
- Can be triggered manually

## 📁 Directory Structure

```
bengali-code-llm/
├── .github/
│   └── workflows/
│       └── train_model.yml
├── scripts/
│   ├── data_collector.py
│   ├── tokenizer_trainer.py
│   ├── model_trainer.py
│   └── model_evaluator.py
├── data/
│   └── raw/
├── outputs/
│   ├── tokenizer/
│   ├── model/
│   └── evaluation/
├── requirements.txt
└── README.md
```

## 🎯 Model Performance

The model is evaluated on various tasks:
- Code generation in Bengali
- Code explanation and documentation
- Error detection and correction
- Algorithm explanation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions and feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- TinyLlama team for the base model
- HuggingFace for the Transformers library
- Weights & Biases for experiment tracking
