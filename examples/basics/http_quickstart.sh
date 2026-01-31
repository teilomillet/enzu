#!/usr/bin/env bash
set -euo pipefail

curl http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Say hello","model":"gpt-4o","provider":"openai"}'
