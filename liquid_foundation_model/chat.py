import argparse
import torch
from transformers import AutoTokenizer

from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.liquid_foundation_model import LiquidFoundationModelForCausalLM

def format_chat_prompt(messages):
    """Format a list of messages into a chat prompt."""
    prompt = "<|startoftext|>"
    
    # Add system message if provided
    if messages[0]["role"] == "system":
        prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        messages = messages[1:]
    
    # Add user/assistant messages
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add the assistant prompt for the response
    prompt += "<|im_start|>assistant\n"
    
    return prompt

def generate_response(model, tokenizer, messages, max_length=100, temperature=0.3, top_p=0.9, top_k=50, min_p=0.15, repetition_penalty=1.05, use_hrm=None):
    """Generate a response from the model based on the conversation history."""
    # Format the prompt
    prompt = format_chat_prompt(messages)
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Detect if this is a reasoning task
    task_type = "reasoning" if any(word in prompt.lower() for word in ["solve", "calculate", "reason", "think", "step by step", "problem"]) else "auto"
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    response = output_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    return response

def chat_loop(model, tokenizer):
    """Interactive chat loop with the model."""
    print("Welcome to the LFM2 chat! Type 'exit' to end the conversation.")
    
    # Initialize conversation with a system message
    conversation = [
        {"role": "system", "content": "You are a helpful assistant trained by Liquid AI."}
    ]
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        
        response = generate_response(model, tokenizer, conversation)
        print(response)
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response})

def main():
    parser = argparse.ArgumentParser(description="Chat with the Liquid Foundation Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for sampling")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = LiquidFoundationModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Make sure we have the necessary special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Start chat loop
    chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()