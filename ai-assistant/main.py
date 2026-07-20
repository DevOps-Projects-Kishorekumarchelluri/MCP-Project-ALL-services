from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import os
import httpx
import json
import requests

app = FastAPI(title="AI Assistant Service", version="2.0.0")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = "qwen2.5-coder:1.5b"

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
        "name": "get_cluster_status",
        "description": "Get live Kubernetes cluster status: pods, nodes, CPU/memory usage, cluster info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Kubernetes namespace (optional, defaults to mcp-platform)"}
            }
        }
    },
    {
        "name": "get_products",
        "description": "List all products in the platform catalog. Optionally filter by category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category (optional)"}
            }
        }
    },
    {
        "name": "get_product",
        "description": "Get details of a specific product by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID"}
            },
            "required": ["product_id"]
        }
    },
    {
        "name": "get_users",
        "description": "List all registered users on the platform.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_models",
        "description": "List AI models registered in the MCP model registry.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_payment",
        "description": "Look up a payment record by payment ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "payment_id": {"type": "string", "description": "The payment UUID"}
            },
            "required": ["payment_id"]
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
    tools_used = []

    # Build messages for Ollama
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        text_response = data.get("message", {}).get("content", "No response")

        return {
            "response": text_response,
            "session_id": req.session_id,
            "tools_used": tools_used,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            }
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "session_id": req.session_id,
            "tools_used": tools_used,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


@app.post("/summarize")
def summarize(text: str, max_tokens: int = 512):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": f"Summarize the following:\n\n{text}"}],
                "stream": False,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        summary = data.get("message", {}).get("content", "No summary")
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"Error: {str(e)}"}
