# MCP-Project-ALL-services


```
#!/bin/bash

ACCOUNT=340640891954
REGION=ap-northeast-1

for svc in mcp-api-gateway auth-service mcp-control-plane model-service ai-assistant recommendation-engine product-service user-service payment-service frontend; do
    aws ecr create-repository \
      --repository-name mcp-platform/$svc \
      --region $REGION \
      --image-scanning-configuration scanOnPush=true
    echo "Created: mcp-platform/$svc"
  done

```