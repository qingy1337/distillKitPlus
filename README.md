# DistillKitPlus

A powerful toolkit for knowledge distillation of Large Language Models (LLMs) using logits.

## Features

- Support for teacher-student knowledge distillation
- Pre-computed logits support for memory-efficient training
- LoRA fine-tuning integration
- Quantization support (4-bit, 8-bit)
- Flash Attention 2 support
- Flexible dataset formatting
- Support for various model architectures

## Installation

```bash
git clone https://github.com/yourusername/distillkitplus.git
cd distillkitplus
pip install -r requirements.txt
```

## Quick Start

1. Configure your distillation settings in `config/default_config.json`
2. Generate teacher logits:
```bash
python scripts/local/generate_logits.py --config config/default_config.json
```
3. Run distillation:
```bash
python scripts/local/distill_logits.py --config config/default_config.json
```

## Configuration

The toolkit uses a JSON configuration file with the following main sections:

- `project_name`: Name of your distillation project
- `dataset`: Dataset configuration including source and processing settings
- `models`: Teacher and student model specifications
- `tokenizer`: Tokenizer settings including max length and padding
- `training`: Training hyperparameters
- `distillation`: Distillation-specific parameters (temperature, alpha)
- `lora`: LoRA configuration for efficient fine-tuning
- `quantization`: Model quantization settings

See `config/default_config.json` for a complete example.

## Key Components

- `components/models.py`: Model loading and configuration
- `components/trainer.py`: Custom trainer for knowledge distillation
- `components/dataset.py`: Dataset handling and processing
- `components/formatters.py`: Data formatting utilities
- `scripts/local/`: Local training scripts

## Features in Detail

### Pre-computed Logits
Save memory by pre-computing teacher logits:
```python
python scripts/local/generate_logits.py --config your_config.json
```

### LoRA Integration
Enable efficient fine-tuning with LoRA by configuring in your JSON:
```json
"lora": {
    "enable_training": true,
    "r": 8,
    "alpha": 16
}
```

### Quantization
Enable 4-bit quantization:
```json
"quantization": {
    "enabled": true
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Accelerate
- PEFT
- TRL
- See `requirements.txt` for complete list

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
[Add your citation information here]
```
