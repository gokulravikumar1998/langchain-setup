from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product_name: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"Looking up the price of {product_name}...")
    catalog = {"laptop": 100, "phone": 200, "tablet"   : 300}
    return f"The price of {product_name} is {catalog.get(product_name, 'Product not found')}."


@tool
def apply_discount(price: float, discount_tier: str) -> str:
    """Apply discount tier to price."""
    print(f"Applying {discount_tier} discount to {price}...")
    disc_tier = {"gold": 0.1, "silver": 0.2, "bronze": 0.3}
    return f"The price of {price} with {disc_tier.get(discount_tier, 'Discount not found')} discount is {price * (1 - disc_tier.get(discount_tier, 0))}."

#---Agent Loop Tool Calling---

@traceable(run_name="agent_loop_tool_calling")
def run_agent(query: str) -> str:
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}",temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    print("question:", query)
    messages = [SystemMessage(content="You are a helpful assistant that can look up product prices and apply discounts. Rules: never assume any product prices use the availables tools to find the product prices and calculate the price with the discount if the discount is not provided, ask the user for the discount tier if not provided and ask the user for the product name if not provided."),
     HumanMessage(content=query)]
    
    for i in range(1, MAX_ITERATIONS + 1):
        response = llm_with_tools.invoke(messages)
        print(f"Iteration {i}: {response.content}")
        
        if not response.tool_calls:
            print("Final answer:", response.content)
            return response.content

        # # The model only sees the tool result if it is appended as a ToolMessage
        # # tied to that call's id. Without this, it calls the tool again every turn.
        # for tool_call in response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        observation = tool_dict[tool_name].invoke(tool_args)
        print(f"Tool result: {observation}")
        messages.append(response)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
        #     tool_result = tool_dict[tool_name].invoke(tool_args)
        #     print(f"Tool result: {tool_result}")
        #     messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))


if __name__ == "__main__":
    print("Starting agent loop tool calling...")

    run_agent("What is the price of a laptop for bronze tier?")