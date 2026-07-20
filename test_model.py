"""
Test NeedleBonsai Model - Verify it works for function calling
"""
import json
import sys
sys.path.insert(0, '/OneNeuralX1Tool')

import torch
from tokenizers import Tokenizer
from liquid_foundation_model.model.needle_bonsai import create_model


# Sample tools (like Needle format - matches training data)
TOOLS = {
    "weather": json.dumps([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."}
                },
                "required": ["location"]
            }
        }
    }]),
    "lights": json.dumps([{
        "type": "function",
        "function": {
            "name": "control_lights",
            "description": "Turn lights on or off in a room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string", "description": "Room name."},
                    "action": {"type": "string", "description": "on or off."}
                },
                "required": ["room", "action"]
            }
        }
    }]),
    "timer": json.dumps([{
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a timer for a duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Time duration."}
                },
                "required": ["duration"]
            }
        }
    }]),
    "search": json.dumps([{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"]
            }
        }
    }]),
    "multi": json.dumps([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Set a timer.",
                "parameters": {"type": "object", "properties": {"duration": {"type": "string"}}, "required": ["duration"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "control_lights",
                "description": "Control lights.",
                "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "action": {"type": "string"}}, "required": ["room", "action"]}
            }
        }
    ]),
}


def load_model():
    """Load trained model."""
    model = create_model()
    ckpt = torch.load('/OneNeuralX1Tool/checkpoints/needle_bonsai_best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded: {model.get_info()['parameters_millions']:.2f}M params")
    return model


def load_tokenizer():
    """Load saved tokenizer."""
    tokenizer = Tokenizer.from_file('/OneNeuralX1Tool/checkpoints/tokenizer.json')
    print(f"Tokenizer loaded: vocab={tokenizer.get_vocab_size()}")
    return tokenizer


def test_query(model, tokenizer, query, tools_str):
    """Test a single query."""
    # Match training format exactly
    encoder_text = f"Query: {query}\nTools: {tools_str}"
    enc_tokens = tokenizer.encode(encoder_text)
    enc_ids = torch.tensor([enc_tokens.ids[:256]], dtype=torch.long)
    
    with torch.no_grad():
        output = model.generate(
            enc_ids,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            bos_token_id=1,
            eos_token_id=2,
        )
    
    # Decode response
    response = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    return response


def main():
    print("=" * 60)
    print("Testing NeedleBonsai Model")
    print("=" * 60)
    
    model = load_model()
    tokenizer = load_tokenizer()
    
    test_cases = [
        ("What's the weather in Paris?", "weather"),
        ("Turn on the lights in the bedroom", "lights"),
        ("Set a timer for 10 minutes", "timer"),
        ("How old is the sun?", "search"),
        ("Turn off lights and set timer for 5 minutes", "multi"),
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    for query, tool_key in test_cases:
        tools_str = TOOLS[tool_key]
        response = test_query(model, tokenizer, query, tools_str)
        
        print(f"\nQuery: {query}")
        print(f"Tools: {tool_key}")
        print(f"Response: {response[:400]}")
        
        # Check if response looks like a tool call
        if "<tool_call>" in response or ("name" in response and "{" in response):
            print("Status: OK (tool call format)")
        else:
            print("Status: NEEDS REVIEW")
    
    print("\n" + "=" * 60)
    print("Testing Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
