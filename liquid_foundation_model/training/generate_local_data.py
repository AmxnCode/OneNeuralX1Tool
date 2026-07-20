"""
Generate function calling training data using free local models.

No API needed - runs entirely on your device.

Usage:
    # Generate synthetic tool calling data
    python generate_local_data.py --num-samples 10000 --output data/tool_calling.jsonl
    
    # Generate from existing datasets
    python generate_local_data.py --source gorilla --output data/gorilla.jsonl
"""

import argparse
import json
import os
import random
from typing import List, Dict
from pathlib import Path


# Tool definitions for synthetic data generation
TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "location": "string - City name",
            "units": "string - celsius or fahrenheit"
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": "string - Search query",
            "num_results": "integer - Number of results to return"
        }
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "to": "string - Recipient email",
            "subject": "string - Email subject",
            "body": "string - Email body"
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event",
        "parameters": {
            "title": "string - Event title",
            "date": "string - Date in YYYY-MM-DD format",
            "time": "string - Time in HH:MM format",
            "duration_minutes": "integer - Duration in minutes"
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price",
        "parameters": {
            "symbol": "string - Stock ticker symbol",
            "currency": "string - Currency code"
        }
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "query": "string - Song or artist name",
            "playlist": "string - Optional playlist name"
        }
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder",
        "parameters": {
            "message": "string - Reminder message",
            "datetime": "string - ISO datetime format"
        }
    },
    {
        "name": "translate_text",
        "description": "Translate text to another language",
        "parameters": {
            "text": "string - Text to translate",
            "target_language": "string - Target language code"
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "expression": "string - Math expression"
        }
    },
    {
        "name": "get_news",
        "description": "Get latest news headlines",
        "parameters": {
            "category": "string - News category",
            "country": "string - Country code"
        }
    }
]

# Sample queries for each tool
QUERY_TEMPLATES = {
    "get_weather": [
        "What's the weather in {city}?",
        "How's the weather looking in {city}?",
        "Tell me the forecast for {city}",
        "Is it raining in {city}?",
        "What's the temperature in {city}?"
    ],
    "search_web": [
        "Search for {topic}",
        "Look up {topic}",
        "Find information about {topic}",
        "What is {topic}?",
        "Search the web for {topic}"
    ],
    "send_email": [
        "Send an email to {person}",
        "Email {person} about {topic}",
        "Compose a message to {person}",
        "Send a message to {person}",
        "Write to {person} about {topic}"
    ],
    "create_calendar_event": [
        "Schedule a meeting about {topic}",
        "Create an event for {topic}",
        "Add {topic} to my calendar",
        "Set up a meeting about {topic}",
        "Block time for {topic}"
    ],
    "get_stock_price": [
        "What's the price of {stock}?",
        "How is {stock} doing?",
        "Check {stock} stock price",
        "Current value of {stock}",
        "Stock price for {stock}"
    ],
    "play_music": [
        "Play {song}",
        "Put on {song}",
        "Start playing {song}",
        "I want to hear {song}",
        "Play some {song}"
    ],
    "set_reminder": [
        "Remind me about {task}",
        "Set a reminder for {task}",
        "Don't forget about {task}",
        "Alert me about {task}",
        "Remind me to {task}"
    ],
    "translate_text": [
        "Translate '{text}' to {language}",
        "How do you say '{text}' in {language}?",
        "Convert this to {language}: {text}",
        "Translation needed: '{text}' to {language}",
        "What's '{text}' in {language}?"
    ],
    "calculate": [
        "Calculate {expression}",
        "What's {expression}?",
        "Compute {expression}",
        "Solve {expression}",
        "What is {expression}?"
    ],
    "get_news": [
        "Show me {category} news",
        "What's happening in {category}?",
        "Latest {category} headlines",
        "News about {category}",
        "Get {category} news from {country}"
    ]
}

# Sample values for templates
SAMPLE_VALUES = {
    "city": ["New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Toronto", "Mumbai"],
    "topic": ["machine learning", "climate change", "space exploration", "AI", "blockchain", "quantum computing"],
    "person": ["john@example.com", "sarah@company.com", "team@work.com"],
    "song": ["Bohemian Rhapsody", "Yesterday", "Shape of You", "Blinding Lights"],
    "stock": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
    "task": ["buy groceries", "call mom", "finish report", "book flight"],
    "text": ["hello", "thank you", "how are you", "good morning"],
    "language": ["Spanish", "French", "German", "Japanese", "Chinese"],
    "expression": ["2+2", "100/5", "15*3", "sqrt(16)", "2^10"],
    "category": ["technology", "sports", "business", "entertainment"],
    "country": ["US", "UK", "Japan", "Germany"],
}


def generate_tool_call(tool: Dict, query: str) -> Dict:
    """Generate a tool call response for a given query."""
    # Extract parameters from query (simplified)
    arguments = {}
    
    if tool["name"] == "get_weather":
        for city in SAMPLE_VALUES["city"]:
            if city.lower() in query.lower():
                arguments = {"location": city, "units": "celsius"}
                break
        if not arguments:
            arguments = {"location": "New York", "units": "celsius"}
    
    elif tool["name"] == "search_web":
        arguments = {"query": query.replace("search for ", "").replace("look up ", ""), "num_results": 5}
    
    elif tool["name"] == "send_email":
        arguments = {"to": "john@example.com", "subject": "Message", "body": query}
    
    elif tool["name"] == "create_calendar_event":
        arguments = {"title": "Meeting", "date": "2024-01-15", "time": "10:00", "duration_minutes": 60}
    
    elif tool["name"] == "get_stock_price":
        arguments = {"symbol": "AAPL", "currency": "USD"}
    
    elif tool["name"] == "play_music":
        arguments = {"query": "Bohemian Rhapsody"}
    
    elif tool["name"] == "set_reminder":
        arguments = {"message": "Don't forget", "datetime": "2024-01-15T10:00:00"}
    
    elif tool["name"] == "translate_text":
        arguments = {"text": "hello", "target_language": "es"}
    
    elif tool["name"] == "calculate":
        arguments = {"expression": "2+2"}
    
    elif tool["name"] == "get_news":
        arguments = {"category": "technology", "country": "us"}
    
    return {
        "name": tool["name"],
        "arguments": arguments
    }


def generate_synthetic_data(num_samples: int = 10000) -> List[Dict]:
    """Generate synthetic function calling data."""
    data = []
    
    for _ in range(num_samples):
        # Pick random tool
        tool = random.choice(TOOL_DEFINITIONS)
        
        # Pick random query template
        template = random.choice(QUERY_TEMPLATES[tool["name"]])
        
        # Fill in template with random values
        query = template
        for key, values in SAMPLE_VALUES.items():
            if "{" + key + "}" in query:
                query = query.replace("{" + key + "}", random.choice(values))
        
        # Generate tool call
        tool_call = generate_tool_call(tool, query)
        
        # Create training example
        example = {
            "query": query,
            "tools": json.dumps([tool], indent=2),
            "response": json.dumps([tool_call], indent=2)
        }
        
        data.append(example)
    
    return data


def download_gorilla_dataset(output_path: str, max_samples: int = 10000):
    """Download and format Gorilla function calling dataset."""
    try:
        from datasets import load_dataset
        
        print("Downloading Gorilla dataset...")
        dataset = load_dataset("gorilla-llm/gorilla-openfunctions-v1", split="train")
        
        data = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            # Format for our training
            example = {
                "query": item.get("question", ""),
                "tools": item.get("api_call", "[]"),
                "response": item.get("api_response", "[]")
            }
            data.append(example)
        
        # Save
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Saved {len(data)} examples to {output_path}")
        
    except Exception as e:
        print(f"Error downloading Gorilla: {e}")
        print("Falling back to synthetic data generation...")


def download_toolbench_dataset(output_path: str, max_samples: int = 10000):
    """Download and format ToolBench dataset."""
    try:
        from datasets import load_dataset
        
        print("Downloading ToolBench dataset...")
        dataset = load_dataset("shuyuej/toolbench", split="train")
        
        data = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            # Format for our training
            example = {
                "query": item.get("query", ""),
                "tools": item.get("tools", "[]"),
                "response": item.get("response", "[]")
            }
            data.append(example)
        
        # Save
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Saved {len(data)} examples to {output_path}")
        
    except Exception as e:
        print(f"Error downloading ToolBench: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate function calling training data")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/tool_calling.jsonl", help="Output file path")
    parser.add_argument("--source", type=str, choices=["synthetic", "gorilla", "toolbench"], 
                        default="synthetic", help="Data source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.source == "synthetic":
        # Generate synthetic data
        print(f"Generating {args.num_samples} synthetic function calling examples...")
        data = generate_synthetic_data(args.num_samples)
        
        # Save
        with open(args.output, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Saved {len(data)} examples to {args.output}")
        
    elif args.source == "gorilla":
        download_gorilla_dataset(args.output, args.num_samples)
        
    elif args.source == "toolbench":
        download_toolbench_dataset(args.output, args.num_samples)
    
    # Print sample
    print("\nSample output:")
    with open(args.output, "r") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            item = json.loads(line)
            print(f"\nQuery: {item['query']}")
            print(f"Tools: {item['tools'][:100]}...")
            print(f"Response: {item['response'][:100]}...")


if __name__ == "__main__":
    main()
