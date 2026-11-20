from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"Looking up the price of {product}...")
    catalog = {"laptop": 100, "phone": 200, "tablet"   : 300}
    return f"The price of {product} is {catalog.get(product, 'Product not found')}."


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> str:
    """Apply discount tier to price."""
    print(f"Applying {discount_tier} discount to {price}...")
    disc_tier = {"gold": 0.1, "silver": 0.2, "bronze": 0.3}
    return f"The price of {price} with {disc_tier.get(discount_tier, 'Discount not found')} discount is {price * (1 - disc_tier.get(discount_tier, 0))}."


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

#---Agent Loop Tool Calling---

@traceable(run_name="ollama_raw_function_calling")
def run_ollama(query: str) -> str:
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=query)

@traceable(run_name="agent_loop_tool_calling")
def run_agent(query: str) -> str:
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print("question:", query)
    messages = [{"role": "system", "content": "You are a helpful assistant that can look up product prices and apply discounts. Rules: never assume any product prices use the availables tools to find the product prices and calculate the price with the discount if the discount is not provided, ask the user for the discount tier if not provided and ask the user for the product name if not provided."},
     {"role": "user", "content": query}]
    
    for i in range(1, MAX_ITERATIONS + 1):
        response = run_ollama(messages)
        ai_message = response.message
        
        if not ai_message.tool_calls:
            print("Final answer:", ai_message.content)
            return ai_message.content

        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        print(f"Tool name: {tool_name}")
        print(f"Tool args: {tool_args}")
        if tool_name not in tools_dict:
            print(f"Tool {tool_name} not found in tools_dict")
            raise ValueError(f"Tool {tool_name} not found in tools_dict")
        tool_result = tools_dict[tool_name](**tool_args)
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(tool_result)})
        print(f"Tool result: {tool_result}")
    return "No answer found"

if __name__ == "__main__":
    print("Starting agent loop raw function calling...")
    run_agent("What is the price of a laptop for bronze tier?")