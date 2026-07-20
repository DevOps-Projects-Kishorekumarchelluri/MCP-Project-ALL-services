from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from anthropic import Anthropic
import os
import httpx
import json
import requests

app = FastAPI(title="AI Assistant Service", version="2.0.0")

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CONTROL_PLANE_URL   = os.getenv("MCP_CONTROL_PLANE_URL",     "http://mcp-control-plane:8008")
PRODUCT_SERVICE_URL = os.getenv("PRODUCT_SERVICE_URL",       "http://product-service:8005")
USER_SERVICE_URL    = os.getenv("USER_SERVICE_URL",          "http://user-service:8006")
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL",       "http://payment-service:8007")
MODEL_SERVICE_URL   = os.getenv("MODEL_SERVICE_URL",         "http://model-service:8002")

SYSTEM_PROMPT = """You are an intelligent AI assistant built into the MCP Platform — a cloud-native microservices platform running on Kubernetes/EKS.

You have two modes:
1. GENERAL ASSISTANT — answer any question: coding, explanations, writing, analysis, maths, etc.
2. MCP PLATFORM ASSISTANT — query live platform data using your tools whenever the user asks about:
   - Products, inventory, catalog
   - Users, accounts, registrations
   - Cluster health, pods, nodes, resource usage
   - AI models in the registry
   - Payments, transactions

Rules:
- Always use tools when the user asks about platform data — never make up platform-specific details.
- For general questions (not about the platform), answer directly from your knowledge.
- Be concise but complete. Format data clearly using lists or tables when helpful.
- If a tool returns an error, tell the user which service is unreachable."""

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cluster_status",
            "description": "Get live Kubernetes cluster status: pods, nodes, CPU/memory usage, cluster info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace (optional, defaults to mcp-platform)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_products",
            "description": "List all products in the platform catalog. Optionally filter by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category (optional)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Get details of a specific product by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product ID"}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_users",
            "description": "List all registered users on the platform.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_models",
            "description": "List AI models registered in the MCP model registry.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_payment",
            "description": "Look up a payment record by payment ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_id": {"type": "string", "description": "The payment UUID"}
                },
                "required": ["payment_id"]
            }
        }
    },
]


def call_tool(name: str, args: dict) -> str:
    try:
        with httpx.Client(timeout=8.0) as http:
            if name == "get_cluster_status":
                params = {"namespace": args["namespace"]} if "namespace" in args else {}
                r = http.get(f"{CONTROL_PLANE_URL}/status", params=params)
                return r.text

            elif name == "get_products":
                params = {"category": args["category"]} if "category" in args else {}
                r = http.get(f"{PRODUCT_SERVICE_URL}/products", params=params)
                return r.text

            elif name == "get_product":
                r = http.get(f"{PRODUCT_SERVICE_URL}/products/{args['product_id']}")
                return r.text

            elif name == "get_users":
                r = http.get(f"{USER_SERVICE_URL}/users")
                return r.text

            elif name == "get_models":
                r = http.get(f"{CONTROL_PLANE_URL}/models")
                return r.text

            elif name == "get_payment":
                r = http.get(f"{PAYMENT_SERVICE_URL}/payments/{args['payment_id']}")
                return r.text

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    max_tokens: Optional[int] = 1024


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-assistant"}


@app.post("/chat")
def chat(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    tools_used = []
    total_input = 0
    total_output = 0

    # Tool-calling loop (max 5 rounds)
    for _ in range(5):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=req.max_tokens,
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,
            messages=msgs,
        )
        total_input  += response.usage.input_tokens
        total_output += response.usage.output_tokens

        # Extract text response
        text_response = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_response = block.text
            elif block.type == "tool_use":
                tool_calls.append(block)

        # No tool calls — final answer
        if not tool_calls or response.stop_reason == "end_turn":
            return {
                "response":   text_response,
                "session_id": req.session_id,
                "tools_used": tools_used,
                "usage": {
                    "input_tokens":  total_input,
                    "output_tokens": total_output,
                }
            }

        # Add assistant response to conversation
        msgs.append({"role": "assistant", "content": response.content})

        # Execute tool calls
        for tc in tool_calls:
            args = {}
            try:
                args = json.loads(tc.input)
            except Exception:
                pass

            result = call_tool(tc.name, args)
            tools_used.append({"tool": tc.name, "args": args})

            msgs.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    }
                ],
            })

    return {
        "response":   "Reached maximum tool-call rounds without a final answer.",
        "session_id": req.session_id,
        "tools_used": tools_used,
        "usage":      {"input_tokens": total_input, "output_tokens": total_output},
    }


@app.post("/summarize")
def summarize(text: str, max_tokens: int = 512):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": f"Summarize the following:\n\n{text}"}],
    )
    summary = ""
    for block in response.content:
        if block.type == "text":
            summary = block.text
            break
    return {"summary": summary}
