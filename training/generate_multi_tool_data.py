"""
Multi-Tool Chain Training Data Generator

Generates training examples with:
1. Single tool calls (easy)
2. Two-tool chains (medium)
3. Three+ tool chains (hard)

Uses Phi-3-mini as teacher for distillation.
"""

import json
import random
import torch
from typing import List, Dict, Tuple
from pathlib import Path

# Tool definitions
TOOLS = {
    "get_weather": {
        "description": "Get current weather for a location",
        "parameters": {
            "location": "City name (e.g., 'Paris', 'Tokyo')",
            "units": "Temperature units: 'celsius' or 'fahrenheit'"
        }
    },
    "search_flights": {
        "description": "Search for flights between cities",
        "parameters": {
            "origin": "Departure city",
            "destination": "Arrival city",
            "date": "Travel date (YYYY-MM-DD)",
            "max_price": "Maximum price in USD"
        }
    },
    "book_flight": {
        "description": "Book a flight",
        "parameters": {
            "flight_id": "Flight identifier from search results",
            "passenger_name": "Full name of passenger",
            "email": "Contact email"
        }
    },
    "create_event": {
        "description": "Create a calendar event",
        "parameters": {
            "title": "Event title",
            "date": "Event date (YYYY-MM-DD)",
            "time": "Event time (HH:MM)",
            "location": "Event location"
        }
    },
    "send_email": {
        "description": "Send an email",
        "parameters": {
            "to": "Recipient email",
            "subject": "Email subject",
            "body": "Email body text"
        }
    },
    "search_knowledge_base": {
        "description": "Search internal knowledge base",
        "parameters": {
            "query": "Search query",
            "category": "Optional category filter"
        }
    },
    "calculate": {
        "description": "Perform mathematical calculation",
        "parameters": {
            "expression": "Mathematical expression to evaluate"
        }
    },
    "get_directions": {
        "description": "Get directions between locations",
        "parameters": {
            "origin": "Starting location",
            "destination": "Ending location",
            "mode": "Travel mode: 'driving', 'walking', 'transit'"
        }
    },
    "set_reminder": {
        "description": "Set a reminder",
        "parameters": {
            "message": "Reminder message",
            "datetime": "When to remind (YYYY-MM-DD HH:MM)"
        }
    },
    "get_stock_price": {
        "description": "Get current stock price",
        "parameters": {
            "symbol": "Stock ticker symbol (e.g., 'AAPL')"
        }
    }
}

# Multi-tool chain templates
CHAIN_TEMPLATES = [
    # Two-tool chains
    {
        "pattern": "weather_then_calendar",
        "templates": [
            "What's the weather in {city}? If it's nice, create an outdoor event for {day}",
            "Check weather in {city} and schedule a {activity} if conditions are good",
            "Is the weather good in {city} on {day}? Add it to my calendar if so"
        ],
        "tools": ["get_weather", "create_event"],
        "chain_logic": "weather → conditional event"
    },
    {
        "pattern": "flight_then_email",
        "templates": [
            "Find flights to {city} and email me the cheapest option",
            "Search flights from {origin} to {city} and send details to {email}",
            "Look up flights to {city} for {date} and share via email"
        ],
        "tools": ["search_flights", "send_email"],
        "chain_logic": "search → share results"
    },
    {
        "pattern": "search_then_directions",
        "templates": [
            "Find a {place} near me and get directions there",
            "Search for {place} in {city} and show me how to get there",
            "Look up {place} locations and give me directions from {origin}"
        ],
        "tools": ["search_knowledge_base", "get_directions"],
        "chain_logic": "find → navigate"
    },
    {
        "pattern": "stock_then_reminder",
        "templates": [
            "Check {stock} price and remind me if it drops below {price}",
            "Get {stock} price and set a reminder to buy at {price}",
            "Look up {stock} and remind me to check it tomorrow"
        ],
        "tools": ["get_stock_price", "set_reminder"],
        "chain_logic": "monitor → alert"
    },
    {
        "pattern": "calculate_then_email",
        "templates": [
            "Calculate {expression} and email me the result",
            "Compute {expression} and send it to {email}",
            "Do the math: {expression} and share via email"
        ],
        "tools": ["calculate", "send_email"],
        "chain_logic": "compute → share"
    },
    # Three-tool chains
    {
        "pattern": "weather_flight_calendar",
        "templates": [
            "What's the weather in {city}? Find flights there and add to my calendar",
            "Check {city} weather, search flights, and create a travel event",
            "Plan a trip to {city}: check weather, find flights, schedule it"
        ],
        "tools": ["get_weather", "search_flights", "create_event"],
        "chain_logic": "research → book → schedule"
    },
    {
        "pattern": "search_directions_email",
        "templates": [
            "Find {place} in {city}, get directions, and email them to me",
            "Search for {place}, calculate route, and send details to {email}",
            "Look up {place} locations, get directions from {origin}, share via email"
        ],
        "tools": ["search_knowledge_base", "get_directions", "send_email"],
        "chain_logic": "find → route → share"
    },
    {
        "pattern": "stock_calculate_reminder",
        "templates": [
            "Check {stock}, calculate my profit if I sell at {price}, remind me",
            "Get {stock} price, compute potential gains, set a reminder",
            "Look up {stock}, do the math on {shares} shares, remind me to sell"
        ],
        "tools": ["get_stock_price", "calculate", "set_reminder"],
        "chain_logic": "monitor → compute → alert"
    },
]


class MultiToolChainGenerator:
    """Generate multi-tool chain training examples."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.locations = [
            "Paris", "Tokyo", "New York", "London", "Sydney",
            "Berlin", "Toronto", "Mumbai", "Dubai", "Singapore"
        ]
        self.places = [
            "restaurant", "coffee shop", "gym", "park", "museum",
            "library", "store", "pharmacy", "bank", "hospital"
        ]
        self.activities = [
            "picnic", "barbecue", "sports game", "concert", "meeting",
            "workout", "study session", "party", "dinner", "lunch"
        ]
        self.stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
        self.emails = [
            "user@example.com", "me@gmail.com", "john@work.com",
            "sarah@company.com", "alex@email.com"
        ]
    
    def generate_single_tool_example(self) -> Dict:
        """Generate a single tool call example."""
        tool_name = random.choice(list(TOOLS.keys()))
        tool = TOOLS[tool_name]
        
        # Generate query and arguments based on tool
        query, arguments = self._generate_tool_call(tool_name)
        
        return {
            "query": query,
            "tools": json.dumps([{"name": tool_name, "description": tool["description"], "parameters": tool["parameters"]}]),
            "answers": json.dumps([{"name": tool_name, "arguments": arguments}])
        }
    
    def generate_chain_example(self, chain_template: Dict) -> Dict:
        """Generate a multi-tool chain example."""
        # Pick a template
        template = random.choice(chain_template["templates"])
        
        # Fill in template
        fills = self._get_template_fills(chain_template["pattern"])
        query = template.format(**fills)
        
        # Generate tool calls
        tool_calls = []
        for tool_name in chain_template["tools"]:
            tool = TOOLS[tool_name]
            _, arguments = self._generate_tool_call(tool_name, context=fills)
            tool_calls.append({"name": tool_name, "arguments": arguments})
        
        # Get all tools for context
        all_tools = [
            {"name": name, "description": tool["description"], "parameters": tool["parameters"]}
            for name, tool in TOOLS.items()
        ]
        
        return {
            "query": query,
            "tools": json.dumps(all_tools),
            "answers": json.dumps(tool_calls)
        }
    
    def _generate_tool_call(self, tool_name: str, context: Dict = None) -> Tuple[str, Dict]:
        """Generate a query and arguments for a specific tool."""
        context = context or {}
        
        if tool_name == "get_weather":
            city = context.get("city", random.choice(self.locations))
            return f"What's the weather in {city}?", {"location": city, "units": "celsius"}
        
        elif tool_name == "search_flights":
            origin = context.get("origin", random.choice(self.locations))
            dest = context.get("city", random.choice([l for l in self.locations if l != origin]))
            date = context.get("date", "2024-03-15")
            return f"Find flights from {origin} to {dest}", {
                "origin": origin, "destination": dest, "date": date
            }
        
        elif tool_name == "book_flight":
            return "Book this flight", {
                "flight_id": f"FL{random.randint(1000, 9999)}",
                "passenger_name": "John Doe",
                "email": random.choice(self.emails)
            }
        
        elif tool_name == "create_event":
            city = context.get("city", random.choice(self.locations))
            activity = context.get("activity", random.choice(self.activities))
            return f"Create a {activity} event", {
                "title": f"{activity.title()} in {city}",
                "date": "2024-03-20",
                "time": "14:00",
                "location": city
            }
        
        elif tool_name == "send_email":
            email = context.get("email", random.choice(self.emails))
            return f"Send email to {email}", {
                "to": email,
                "subject": "Flight Information",
                "body": "Here are the details you requested."
            }
        
        elif tool_name == "search_knowledge_base":
            place = context.get("place", random.choice(self.places))
            city = context.get("city", random.choice(self.locations))
            return f"Search for {place} in {city}", {
                "query": f"{place} in {city}",
                "category": "locations"
            }
        
        elif tool_name == "calculate":
            expr = random.choice(["2+2", "15*3", "100/4", "50-25", "2**10"])
            return f"Calculate {expr}", {"expression": expr}
        
        elif tool_name == "get_directions":
            origin = context.get("origin", "Current location")
            dest = context.get("city", random.choice(self.locations))
            return f"Directions to {dest}", {
                "origin": origin,
                "destination": dest,
                "mode": random.choice(["driving", "walking", "transit"])
            }
        
        elif tool_name == "set_reminder":
            msg = context.get("message", "Check stock price")
            return f"Set reminder: {msg}", {
                "message": msg,
                "datetime": "2024-03-21 09:00"
            }
        
        elif tool_name == "get_stock_price":
            stock = context.get("stock", random.choice(self.stocks))
            return f"Get {stock} price", {"symbol": stock}
        
        return "Do something", {}
    
    def _get_template_fills(self, pattern: str) -> Dict:
        """Get random fills for a template pattern."""
        city = random.choice(self.locations)
        origin = random.choice([l for l in self.locations if l != city])
        
        fills = {
            "city": city,
            "origin": origin,
            "day": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
            "date": "2024-03-15",
            "activity": random.choice(self.activities),
            "place": random.choice(self.places),
            "stock": random.choice(self.stocks),
            "price": str(random.randint(50, 500)),
            "email": random.choice(self.emails),
            "expression": random.choice(["2+2", "15*3", "100/4"]),
            "shares": str(random.randint(10, 100))
        }
        return fills
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        single_tool_ratio: float = 0.3,
        two_tool_ratio: float = 0.4,
        three_tool_ratio: float = 0.3,
    ) -> List[Dict]:
        """
        Generate balanced dataset with single, 2-tool, and 3-tool chains.
        
        Args:
            num_samples: Total number of samples
            single_tool_ratio: Fraction of single tool calls
            two_tool_ratio: Fraction of 2-tool chains
            three_tool_ratio: Fraction of 3-tool chains
            
        Returns:
            List of training examples
        """
        examples = []
        
        # Single tool calls
        num_single = int(num_samples * single_tool_ratio)
        for _ in range(num_single):
            examples.append(self.generate_single_tool_example())
        
        # Two-tool chains
        two_tool_chains = [c for c in CHAIN_TEMPLATES if len(c["tools"]) == 2]
        num_two = int(num_samples * two_tool_ratio)
        for _ in range(num_two):
            chain = random.choice(two_tool_chains)
            examples.append(self.generate_chain_example(chain))
        
        # Three-tool chains
        three_tool_chains = [c for c in CHAIN_TEMPLATES if len(c["tools"]) == 3]
        num_three = num_samples - num_single - num_two
        for _ in range(num_three):
            chain = random.choice(three_tool_chains)
            examples.append(self.generate_chain_example(chain))
        
        # Shuffle
        random.shuffle(examples)
        
        return examples
    
    def save_dataset(self, examples: List[Dict], output_path: str):
        """Save dataset to JSONL file."""
        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        print(f"Saved {len(examples)} examples to {output_path}")


def main():
    """Generate training dataset."""
    generator = MultiToolChainGenerator(seed=42)
    
    # Generate dataset
    print("Generating multi-tool chain dataset...")
    examples = generator.generate_dataset(
        num_samples=5000,
        single_tool_ratio=0.3,
        two_tool_ratio=0.4,
        three_tool_ratio=0.3,
    )
    
    # Count by type
    single = sum(1 for e in examples if json.loads(e["answers"]).__len__() == 1)
    two = sum(1 for e in examples if json.loads(e["answers"]).__len__() == 2)
    three = sum(1 for e in examples if json.loads(e["answers"]).__len__() >= 3)
    
    print(f"\nDataset breakdown:")
    print(f"  Single tool: {single} ({single/len(examples)*100:.1f}%)")
    print(f"  2-tool chain: {two} ({two/len(examples)*100:.1f}%)")
    print(f"  3-tool chain: {three} ({three/len(examples)*100:.1f}%)")
    
    # Show examples
    print("\nExamples:")
    for i in range(3):
        ex = examples[i]
        print(f"\n{i+1}. Query: {ex['query']}")
        answers = json.loads(ex['answers'])
        print(f"   Tools called: {[a['name'] for a in answers]}")
    
    # Save
    output_path = "/Users/amanpreetsingh/projects/experiment/oneaimodel/data/multi_tool_chains.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    generator.save_dataset(examples, output_path)


if __name__ == "__main__":
    main()
