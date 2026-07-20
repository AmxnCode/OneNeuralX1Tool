# One Neural X1 Model

This repository contains the One Neural X1 model, a hybrid architecture that combines convolution and attention mechanisms for natural language processing.

## Model Details

- **Model Type:** One Neural X1
- **Architecture:** Hybrid architecture with convolution and attention blocks
- **Parameters:** 27.73 million
- **Training Data:** WikiText-2
- **Training Objective:** Causal Language Modeling

## Usage

### Installation

```bash
pip install torch transformers
```

### Loading the Model

```python
import torch
from transformers import GPT2TokenizerFast

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("oneconscious-ai/one-neural-x1")

# Load model weights
model_path = "oneconscious-ai/one-neural-x1"
model_weights = torch.load("model.pt", map_location=torch.device("cpu"))

# Create model configuration
from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModelForCausalLM

config = LFMConfig(
    model_size="small",
    num_layers=6,
    hidden_size=384,
    intermediate_size=1536,
    num_attention_heads=6,
    num_key_value_heads=3,
    max_position_embeddings=256,
    vocab_size=len(tokenizer),
    conv_kernel_size=3,
    num_conv_blocks=3,
    num_attention_blocks=3,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
)

# Create model and load weights
model = LiquidFoundationModelForCausalLM(config)
model.load_state_dict(model_weights)
model.eval()
```

### Generating Text

```python
# Prepare input
prompt = "The quick brown fox"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output_ids = model.generate(
    input_ids=input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)

# Decode the output
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
```

## Limitations

This model is a research prototype and has several limitations:

- Limited training data (WikiText-2)
- Relatively small model size (27.73M parameters)
- May generate repetitive or nonsensical text
- Limited context understanding

## Future Work

- Train on larger and more diverse datasets
- Increase model size for better performance
- Fine-tune for specific tasks
- Improve the hybrid architecture

## Citation

```
@misc{one-neural-x1-model,
  author = {OneConscious AI},
  title = {One Neural X1: A Hybrid Architecture for Language Processing},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/oneconscious-ai/one-neural-x1}}
}
```