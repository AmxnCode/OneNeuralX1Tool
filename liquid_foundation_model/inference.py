import argparse
import torch
from transformers import AutoTokenizer

from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModelForCausalLM

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.3,
    top_p=0.9,
    top_k=50,
    min_p=0.15,
    repetition_penalty=1.05,
    do_sample=True,
    num_return_sequences=1,
):
    """Generate text using the model."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
    
    # Decode and return generated text
    generated_texts = []
    for i in range(num_return_sequences):
        generated_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

def main():
    parser = argparse.ArgumentParser(description="Generate text with the Liquid Foundation Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for text generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--min-p", type=float, default=0.15, help="Min-p sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--num-sequences", type=int, default=1, help="Number of sequences to generate")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = LiquidFoundationModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Generate text
    print(f"Generating text with prompt: '{args.prompt}'")
    generated_texts = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_sequences,
    )
    
    # Print generated text
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated text {i+1}:")
        print(text)

if __name__ == "__main__":
    main()