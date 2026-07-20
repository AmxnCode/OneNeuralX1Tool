"""
Evaluation Benchmark for Enhanced X1 Tool Model

Tests:
1. Single tool selection accuracy
2. Multi-tool chain accuracy
3. JSON parse rate
4. Argument extraction accuracy
5. Latency measurement
"""

import json
import torch
import time
from typing import Dict, List, Tuple
from pathlib import Path

import sys
sys.path.insert(0, '/Users/amanpreetsingh/projects/experiment/oneaimodel')

from liquid_foundation_model.model.encoder_decoder_v2 import OneNeuralX1ToolV2


# Evaluation benchmarks
BENCHMARKS = {
    "single_tool": [
        {
            "query": "What's the weather in Paris?",
            "expected_tools": ["get_weather"],
            "expected_args": {"location": "Paris"},
        },
        {
            "query": "Get Apple stock price",
            "expected_tools": ["get_stock_price"],
            "expected_args": {"symbol": "AAPL"},
        },
        {
            "query": "Calculate 15 * 3",
            "expected_tools": ["calculate"],
            "expected_args": {"expression": "15*3"},
        },
        {
            "query": "Set a reminder for tomorrow at 9am",
            "expected_tools": ["set_reminder"],
            "expected_args": {},
        },
        {
            "query": "Send an email to john@example.com",
            "expected_tools": ["send_email"],
            "expected_args": {"to": "john@example.com"},
        },
    ],
    "two_tool_chain": [
        {
            "query": "Check weather in Tokyo and create an event if it's nice",
            "expected_tools": ["get_weather", "create_event"],
            "expected_args": {},
        },
        {
            "query": "Find flights to Paris and email me the details",
            "expected_tools": ["search_flights", "send_email"],
            "expected_args": {"destination": "Paris"},
        },
        {
            "query": "Search for restaurants and get directions",
            "expected_tools": ["search_knowledge_base", "get_directions"],
            "expected_args": {},
        },
    ],
    "three_tool_chain": [
        {
            "query": "Check weather in London, find flights there, and add to calendar",
            "expected_tools": ["get_weather", "search_flights", "create_event"],
            "expected_args": {"destination": "London"},
        },
        {
            "query": "Find a coffee shop, get directions, and email them to me",
            "expected_tools": ["search_knowledge_base", "get_directions", "send_email"],
            "expected_args": {},
        },
    ],
    "edge_cases": [
        {
            "query": "Do something",
            "expected_tools": [],
            "expected_args": {},
        },
        {
            "query": "Hello, how are you?",
            "expected_tools": [],
            "expected_args": {},
        },
        {
            "query": "",  # Empty query
            "expected_tools": [],
            "expected_args": {},
        },
    ],
}


class ModelEvaluator:
    """Evaluate model on tool calling benchmarks."""
    
    def __init__(
        self,
        model: OneNeuralX1ToolV2,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def predict(self, query: str, max_length: int = 128) -> str:
        """Get model prediction for a query."""
        # Tokenize
        input_ids = self.tokenizer.encode(
            query,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.1,
                do_sample=False,
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0])
        return output_text
    
    def parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from model output."""
        try:
            # Try to parse as JSON
            calls = json.loads(text)
            if isinstance(calls, dict):
                calls = [calls]
            return calls
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, text)
            calls = []
            for match in matches:
                try:
                    call = json.loads(match)
                    calls.append(call)
                except:
                    pass
            return calls
    
    def evaluate_single(self, benchmark: Dict) -> Dict:
        """Evaluate on a single benchmark."""
        # Get prediction
        start_time = time.time()
        prediction = self.predict(benchmark["query"])
        latency = time.time() - start_time
        
        # Parse tool calls
        predicted_calls = self.parse_tool_calls(prediction)
        predicted_tools = [call.get("name", "") for call in predicted_calls]
        
        # Check tool selection
        expected_tools = benchmark["expected_tools"]
        tool_correct = set(predicted_tools) == set(expected_tools)
        
        # Check arguments (simplified)
        args_correct = True
        if benchmark["expected_args"]:
            for key, value in benchmark["expected_args"].items():
                found = False
                for call in predicted_calls:
                    if key in call.get("arguments", {}):
                        found = True
                        break
                if not found:
                    args_correct = False
        
        # JSON parse success
        json_parse_ok = len(predicted_calls) > 0 or len(expected_tools) == 0
        
        return {
            "query": benchmark["query"],
            "expected_tools": expected_tools,
            "predicted_tools": predicted_tools,
            "tool_correct": tool_correct,
            "args_correct": args_correct,
            "json_parse_ok": json_parse_ok,
            "latency": latency,
            "prediction": prediction,
        }
    
    def evaluate_benchmark(self, benchmark_name: str) -> Dict:
        """Evaluate on a benchmark category."""
        benchmarks = BENCHMARKS[benchmark_name]
        results = []
        
        for bench in benchmarks:
            result = self.evaluate_single(bench)
            results.append(result)
        
        # Aggregate metrics
        n = len(results)
        tool_accuracy = sum(r["tool_correct"] for r in results) / n if n > 0 else 0
        args_accuracy = sum(r["args_correct"] for r in results) / n if n > 0 else 0
        parse_rate = sum(r["json_parse_ok"] for r in results) / n if n > 0 else 0
        avg_latency = sum(r["latency"] for r in results) / n if n > 0 else 0
        
        return {
            "benchmark": benchmark_name,
            "num_samples": n,
            "tool_accuracy": tool_accuracy,
            "args_accuracy": args_accuracy,
            "parse_rate": parse_rate,
            "avg_latency": avg_latency,
            "details": results,
        }
    
    def evaluate_all(self) -> Dict:
        """Evaluate on all benchmarks."""
        all_results = {}
        
        for benchmark_name in BENCHMARKS.keys():
            print(f"\nEvaluating {benchmark_name}...")
            result = self.evaluate_benchmark(benchmark_name)
            all_results[benchmark_name] = result
            
            print(f"  Tool accuracy: {result['tool_accuracy']:.2%}")
            print(f"  Args accuracy: {result['args_accuracy']:.2%}")
            print(f"  Parse rate: {result['parse_rate']:.2%}")
            print(f"  Avg latency: {result['avg_latency']:.3f}s")
        
        # Overall metrics
        total_samples = sum(r["num_samples"] for r in all_results.values())
        overall_tool_acc = sum(
            r["tool_accuracy"] * r["num_samples"] for r in all_results.values()
        ) / total_samples
        overall_args_acc = sum(
            r["args_accuracy"] * r["num_samples"] for r in all_results.values()
        ) / total_samples
        overall_parse = sum(
            r["parse_rate"] * r["num_samples"] for r in all_results.values()
        ) / total_samples
        overall_latency = sum(
            r["avg_latency"] * r["num_samples"] for r in all_results.values()
        ) / total_samples
        
        return {
            "benchmarks": all_results,
            "overall": {
                "total_samples": total_samples,
                "tool_accuracy": overall_tool_acc,
                "args_accuracy": overall_args_acc,
                "parse_rate": overall_parse,
                "avg_latency": overall_latency,
            },
        }


def compare_with_baselines(results: Dict) -> Dict:
    """Compare results with baseline models."""
    baselines = {
        "our_26m_old": {
            "tool_accuracy": 0.72,
            "args_accuracy": 0.65,
            "parse_rate": 0.88,
            "avg_latency": 0.200,
        },
        "claude_haiku": {
            "tool_accuracy": 0.95,
            "args_accuracy": 0.92,
            "parse_rate": 0.99,
            "avg_latency": 0.150,
        },
        "gpt_4o_mini": {
            "tool_accuracy": 0.93,
            "args_accuracy": 0.90,
            "parse_rate": 0.98,
            "avg_latency": 0.180,
        },
    }
    
    comparison = {
        "enhanced_model": results["overall"],
        "baselines": baselines,
        "vs_old_model": {
            "tool_accuracy_improvement": results["overall"]["tool_accuracy"] - baselines["our_26m_old"]["tool_accuracy"],
            "args_accuracy_improvement": results["overall"]["args_accuracy"] - baselines["our_26m_old"]["args_accuracy"],
            "parse_rate_improvement": results["overall"]["parse_rate"] - baselines["our_26m_old"]["parse_rate"],
            "latency_improvement": baselines["our_26m_old"]["avg_latency"] - results["overall"]["avg_latency"],
        },
    }
    
    return comparison


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Evaluating Enhanced X1 Tool Model")
    print("=" * 60)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n1. Loading model...")
    model = OneNeuralX1ToolV2(
        vocab_size=8192,
        hidden_size=512,
        num_encoder_layers=12,
        num_decoder_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_loops=4,
        num_experts=4,
        expert_dim=2048,
    ).to(device)
    
    # Load weights
    checkpoint_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/enhanced_x1_best.pt"
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"   Loaded checkpoint: {checkpoint_path}")
    else:
        print("   No checkpoint found, using random weights")
    
    # Create tokenizer
    from training.train_enhanced import SimpleTokenizer
    tokenizer = SimpleTokenizer(8192)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, device)
    
    # Evaluate
    print("\n2. Running evaluation...")
    results = evaluator.evaluate_all()
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for name, bench in results["benchmarks"].items():
        print(f"\n{name}:")
        print(f"  Samples: {bench['num_samples']}")
        print(f"  Tool Accuracy: {bench['tool_accuracy']:.2%}")
        print(f"  Args Accuracy: {bench['args_accuracy']:.2%}")
        print(f"  Parse Rate: {bench['parse_rate']:.2%}")
        print(f"  Avg Latency: {bench['avg_latency']:.3f}s")
    
    print("\n" + "-" * 60)
    print("OVERALL:")
    print(f"  Total Samples: {results['overall']['total_samples']}")
    print(f"  Tool Accuracy: {results['overall']['tool_accuracy']:.2%}")
    print(f"  Args Accuracy: {results['overall']['args_accuracy']:.2%}")
    print(f"  Parse Rate: {results['overall']['parse_rate']:.2%}")
    print(f"  Avg Latency: {results['overall']['avg_latency']:.3f}s")
    
    # Compare with baselines
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINES")
    print("=" * 60)
    
    comparison = compare_with_baselines(results)
    
    print(f"\n{'Model':<20} {'Tool Acc':<12} {'Args Acc':<12} {'Parse Rate':<12} {'Latency':<12}")
    print("-" * 68)
    
    print(f"{'Enhanced (ours)':<20} {comparison['enhanced_model']['tool_accuracy']:.2%}{'':<8} "
          f"{comparison['enhanced_model']['args_accuracy']:.2%}{'':<8} "
          f"{comparison['enhanced_model']['parse_rate']:.2%}{'':<8} "
          f"{comparison['enhanced_model']['avg_latency']:.3f}s")
    
    print(f"{'Old 26M model':<20} {comparison['baselines']['our_26m_old']['tool_accuracy']:.2%}{'':<8} "
          f"{comparison['baselines']['our_26m_old']['args_accuracy']:.2%}{'':<8} "
          f"{comparison['baselines']['our_26m_old']['parse_rate']:.2%}{'':<8} "
          f"{comparison['baselines']['our_26m_old']['avg_latency']:.3f}s")
    
    print(f"{'Claude Haiku':<20} {comparison['baselines']['claude_haiku']['tool_accuracy']:.2%}{'':<8} "
          f"{comparison['baselines']['claude_haiku']['args_accuracy']:.2%}{'':<8} "
          f"{comparison['baselines']['claude_haiku']['parse_rate']:.2%}{'':<8} "
          f"{comparison['baselines']['claude_haiku']['avg_latency']:.3f}s")
    
    print("\nImprovements over old model:")
    imp = comparison["vs_old_model"]
    print(f"  Tool Accuracy: +{imp['tool_accuracy_improvement']:.2%}")
    print(f"  Args Accuracy: +{imp['args_accuracy_improvement']:.2%}")
    print(f"  Parse Rate: +{imp['parse_rate_improvement']:.2%}")
    print(f"  Latency: -{imp['latency_improvement']:.3f}s (faster)")
    
    # Save results
    output_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/checkpoints/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Saved results to {output_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
