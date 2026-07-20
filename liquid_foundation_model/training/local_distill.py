"""
Local distillation script - No API needed.

Uses Phi-3-mini (3.8B) as the primary teacher model.
Phi-3-mini is Microsoft's small but powerful model, excellent at reasoning.

Usage:
    # Distill from Phi-3-mini (recommended, ~4GB RAM)
    python local_distill.py --data data/tool_calling.jsonl
    
    # Distill from Phi-3-mini with custom settings
    python local_distill.py --data data/tool_calling.jsonl --num-samples 5000 --batch-size 4
    
    # Distill from other models (if needed)
    python local_distill.py --teacher qwen2.5-0.5b --data data/tool_calling.jsonl
"""

import argparse
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


# Available free teacher models
TEACHER_MODELS = {
    "phi-3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft's small model, good at reasoning (RECOMMENDED)",
        "ram_required": "4GB",
        "max_length": 4096,
        "torch_dtype": "float16",
    },
    "phi-3-mini-128k": {
        "model_id": "microsoft/Phi-3-mini-128k-instruct",
        "description": "Phi-3-mini with 128K context (needs more RAM)",
        "ram_required": "8GB",
        "max_length": 131072,
        "torch_dtype": "float16",
    },
    "qwen2.5-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Small but capable, runs on any device",
        "ram_required": "1GB",
        "max_length": 8192,
        "torch_dtype": "float16",
    },
    "qwen2.5-1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Good balance of size and capability",
        "ram_required": "2GB",
        "max_length": 8192,
        "torch_dtype": "float16",
    },
    "gemma-2b": {
        "model_id": "google/gemma-2-2b-it",
        "description": "Google's small model",
        "ram_required": "3GB",
        "max_length": 8192,
        "torch_dtype": "float16",
    },
    "smollm2-360m": {
        "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "description": "Very small, fast inference",
        "ram_required": "0.5GB",
        "max_length": 8192,
        "torch_dtype": "float32",
    },
}


class FunctionCallDataset(Dataset):
    """Dataset for function calling data."""
    
    def __init__(self, data_path: str, max_length: int = 2048):
        self.data = []
        self.max_length = max_length
        
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item.get("query", ""),
            "tools": item.get("tools", "[]"),
            "response": item.get("response", "[]"),
        }


class LocalTeacher:
    """Local teacher model for distillation."""
    
    def __init__(self, model_name: str = "phi-3-mini", device: Optional[str] = None):
        self.model_name = model_name
        self.model_info = TEACHER_MODELS.get(model_name)
        
        if not self.model_info:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(TEACHER_MODELS.keys())}")
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                       "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Loading {model_name} on {self.device}...")
        print(f"RAM required: {self.model_info['ram_required']}")
        self._load_model()
    
    def _load_model(self):
        """Load the teacher model with optimizations."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Phi-3 specific optimizations
            load_kwargs = {
                "trust_remote_code": True,
            }
            
            # Use appropriate dtype
            if self.model_info["torch_dtype"] == "float16":
                if self.device.type == "cuda":
                    # Use bitsandbytes for quantization if available
                    try:
                        import bitsandbytes as bnb
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        load_kwargs["quantization_config"] = quantization_config
                        print("Using 4-bit quantization for efficiency")
                    except ImportError:
                        load_kwargs["torch_dtype"] = torch.float16
                        print("Using float16 (install bitsandbytes for 4-bit quantization)")
                else:
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float32
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_info["model_id"],
                trust_remote_code=True
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if self.device.type == "cpu":
                load_kwargs["torch_dtype"] = torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_info["model_id"],
                    **load_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_info["model_id"],
                    device_map="auto",
                    **load_kwargs
                )
            
            print(f"Loaded {self.model_name} successfully")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nMake sure you have the required packages:")
            print("pip install transformers torch accelerate bitsandbytes")
            raise
    
    def generate_tool_call(self, query: str, tools: str) -> str:
        """Generate a tool call response using the teacher model."""
        # Phi-3 specific prompt format
        if "phi-3" in self.model_name.lower():
            prompt = f"""<|system|>
You are a helpful assistant that can use tools to answer questions. You must respond with valid JSON containing tool calls.
<|user|>
Available tools:
{tools}

User query: {query}

Please respond with a JSON array containing the tool call(s) you would make.
If no tools are needed, respond with an empty array [].
<|end|>
<|assistant|>
Tool call:"""
        else:
            prompt = f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tools}

User query: {query}

Please respond with a JSON array containing the tool call(s) you would make.
If no tools are needed, respond with an empty array [].

Tool call:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.model_info["max_length"],
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with Phi-3 optimized settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False,
            )
        
        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_batch(self, queries: List[str], tools_list: List[str]) -> List[str]:
        """Generate tool calls for a batch of queries."""
        responses = []
        for query, tools in zip(queries, tools_list):
            response = self.generate_tool_call(query, tools)
            responses.append(response)
        return responses
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": self.model_name,
            "model_id": self.model_info["model_id"],
            "ram_required": self.model_info["ram_required"],
            "max_length": self.model_info["max_length"],
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()) / 1e6,
        }


def distill_to_x1_tool(
    teacher: LocalTeacher,
    student_model,
    train_data: List[Dict],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
):
    """Distill knowledge from teacher to student."""
    from torch.optim import AdamW
    from torch.cuda.amp import autocast, GradScaler
    
    device = next(student_model.parameters()).device
    
    # Create optimizer
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    # Training loop
    print(f"\nStarting distillation for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Process in batches
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = train_data[i:i + batch_size]
            
            # Get teacher responses
            queries = [item["query"] for item in batch]
            tools_list = [item["tools"] for item in batch]
            
            teacher_responses = teacher.generate_batch(queries, tools_list)
            
            # Prepare student inputs
            # TODO: Tokenize and run student model
            # For now, just print progress
            
        print(f"Epoch {epoch + 1} complete")
    
    print("Distillation complete!")


def main():
    parser = argparse.ArgumentParser(description="Local distillation for X1 Tool")
    parser.add_argument("--teacher", type=str, default="phi-3-mini",
                        choices=list(TEACHER_MODELS.keys()),
                        help="Teacher model to use (default: phi-3-mini)")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--output", type=str, default="data/distilled_responses.jsonl",
                        help="Output path for distilled data")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to distill")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save every N samples")
    
    args = parser.parse_args()
    
    # Print available models
    print("\n" + "="*60)
    print("Available teacher models:")
    print("="*60)
    for name, info in TEACHER_MODELS.items():
        marker = " (RECOMMENDED)" if name == "phi-3-mini" else ""
        print(f"  {name}: {info['description']} (RAM: {info['ram_required']}){marker}")
    print("="*60)
    
    # Load teacher
    print(f"\nLoading {args.teacher}...")
    teacher = LocalTeacher(args.teacher, args.device)
    
    # Print model info
    model_info = teacher.get_model_info()
    print(f"\nModel loaded:")
    print(f"  Name: {model_info['name']}")
    print(f"  Parameters: {model_info['parameters']:.1f}M")
    print(f"  Device: {model_info['device']}")
    print(f"  Max length: {model_info['max_length']}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    with open(args.data, "r") as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples")
    
    # Generate distilled responses
    print(f"\nGenerating distilled responses for {args.num_samples} examples...")
    
    distilled_data = []
    
    for i, item in enumerate(tqdm(data[:args.num_samples])):
        query = item["query"]
        tools = item["tools"]
        
        try:
            # Get teacher response
            teacher_response = teacher.generate_tool_call(query, tools)
            
            # Create distilled example
            distilled_example = {
                "query": query,
                "tools": tools,
                "teacher_response": teacher_response,
                "original_response": item.get("response", ""),
                "teacher_model": args.teacher,
            }
            
            distilled_data.append(distilled_example)
            
            # Save periodically
            if (i + 1) % args.save_every == 0:
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                with open(args.output, "w") as f:
                    for d in distilled_data:
                        f.write(json.dumps(d) + "\n")
                print(f"\nSaved {len(distilled_data)} examples")
            
            # Print sample every 50
            if (i + 1) % 50 == 0:
                print(f"\nSample {i+1}:")
                print(f"  Query: {query[:50]}...")
                print(f"  Teacher: {teacher_response[:100]}...")
                
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # Final save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for d in distilled_data:
            f.write(json.dumps(d) + "\n")
    
    print(f"\nDone! Saved {len(distilled_data)} distilled examples to {args.output}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total processed: {args.num_samples}")
    print(f"  Successful: {len(distilled_data)}")
    print(f"  Failed: {args.num_samples - len(distilled_data)}")
    
    # Print samples
    print("\nSample distilled responses:")
    for i in range(min(3, len(distilled_data))):
        item = distilled_data[i]
        print(f"\n{i+1}. Query: {item['query']}")
        print(f"   Teacher: {item['teacher_response'][:150]}...")


if __name__ == "__main__":
    main()
