import os
import sys
import json
import logging
import requests

from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


# -------------------------------------------------
# Load env vars safely
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# LLM Setup (OpenRouter)
# -------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    temperature=0,
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

# -------------------------------------------------
# Typed State for LangGraph
# -------------------------------------------------
class AgentState(TypedDict):
    user_query: str
    intent: dict
    raw_orders: str
    parsed_orders: List[dict]
    final_orders: List[dict]

# -------------------------------------------------
# Pydantic Schemas (LLM CONSTRAINTS)
# -------------------------------------------------
class IntentSchema(BaseModel):
    state: Optional[str]
    min_total: Optional[float]

class OrderSchema(BaseModel):
    orderId: Optional[str]
    buyer: Optional[str]
    state: Optional[str]
    total: Optional[float]

class OrdersListSchema(BaseModel):
    orders: List[OrderSchema]

# -------------------------------------------------
# Deterministic normalization helpers
# -------------------------------------------------
STATE_NORMALIZATION = {
    "ohio": "OH",
    "california": "CA",
    "new york": "NY",
    # add additional states in a full release, but this covers the dummy_customer_api
}

def normalize_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    return STATE_NORMALIZATION.get(v, value.upper())

def normalize_total(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", ""))
        except ValueError:
            return None
    return None

# -------------------------------------------------
# Node 1: Parse user intent (LLM)
# -------------------------------------------------
def parse_intent(state: AgentState):
    logger.info("Parsing user intent")

    prompt = f"""
Extract filtering intent from the user query.

Rules:
- Only extract fields explicitly mentioned
- Do not infer or guess
- If missing, return null
- Output valid JSON only

Schema:
{{
  "state": string | null,
  "min_total": number | null
}}

User query:
{state["user_query"]}
"""

    response = llm.invoke(prompt)

    try:
        intent = IntentSchema.model_validate_json(response.content).model_dump()
    except ValidationError:
        logger.warning("Intent parsing failed, using empty intent")
        intent = {"state": None, "min_total": None}

    return {"intent": intent}

# -------------------------------------------------
# Node 2: Fetch raw orders from API
# -------------------------------------------------
def fetch_orders(state: AgentState):
    logger.info("Fetching orders from API")

    try:
        r = requests.get("http://localhost:5001/api/orders", timeout=5)
        r.raise_for_status()
        return {"raw_orders": r.text}
    except Exception as e:
        logger.error(f"API error: {e}")
        return {"raw_orders": ""}

# -------------------------------------------------
# Helper: Chunk raw text (context safety)
# -------------------------------------------------
def chunk_text(text: str, max_chars: int = 1500):
    chunks = []
    current = ""

    for line in text.splitlines():
        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += "\n" + line

    if current:
        chunks.append(current)

    return chunks

# -------------------------------------------------
# Node 3: Extract structured orders (LLM)
# -------------------------------------------------
def extract_orders(state: AgentState):
    logger.info("Extracting structured orders")

    raw_text = state["raw_orders"]
    chunks = chunk_text(raw_text)

    extracted_orders = []

    for chunk in chunks:
        prompt = f"""
You extract order data from unstructured text.

Rules:
- Extract only explicitly stated fields
- Do not infer missing values
- Return valid JSON ONLY
- If a field is missing, return null

Schema:
{{
  "orders": [
    {{
      "orderId": string | null,
      "buyer": string | null,
      "state": string | null,
      "total": number | null
    }}
  ]
}}

Text:
{chunk}
"""

        response = llm.invoke(prompt)

        try:
            parsed = OrdersListSchema.model_validate_json(response.content)
            extracted_orders.extend(o.model_dump() for o in parsed.orders)
        except ValidationError:
            logger.warning("Order extraction failed for chunk")

    return {"parsed_orders": extracted_orders}

# -------------------------------------------------
# Node 4: Deterministic filtering & validation
# -------------------------------------------------
def filter_orders(state: AgentState):
    logger.info("Filtering orders deterministically")

    intent = state["intent"]
    results = []

    intent_state = normalize_state(intent.get("state"))

    for order in state["parsed_orders"]:
        if not order.get("orderId"):
            continue

        total = normalize_total(order.get("total"))
        if total is None:
            continue

        order_state = normalize_state(order.get("state"))

        # State filter
        if intent_state:
            if order_state != intent_state:
                continue

        # Total filter
        if intent.get("min_total") is not None:
            if total <= intent["min_total"]:
                continue

        # Write normalized values back
        order["state"] = order_state
        order["total"] = float(f"{total:.2f}")  # pad to 2 decimal places

        results.append(order)

    return {"final_orders": results}

# -------------------------------------------------
# Build LangGraph
# -------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_intent", parse_intent)
    graph.add_node("fetch_orders", fetch_orders)
    graph.add_node("extract_orders", extract_orders)
    graph.add_node("filter_orders", filter_orders)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "fetch_orders")
    graph.add_edge("fetch_orders", "extract_orders")
    graph.add_edge("extract_orders", "filter_orders")

    return graph.compile()

# -------------------------------------------------
# Main function
# -------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: raft-agent \"natural language query\"")
        sys.exit(1)

    query = sys.argv[1]
    app = build_graph()

    final_state = app.invoke({
        "user_query": query,
        "intent": {},
        "raw_orders": "",
        "parsed_orders": [],
        "final_orders": []
    })

    print(json.dumps({"orders": final_state["final_orders"]}, indent=2))

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    main()
