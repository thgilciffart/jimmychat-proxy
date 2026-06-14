#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PORT=6767
BASE="http://localhost:${PORT}"

echo "=== Starting jimmychat-proxy ==="
node index.js &
PID=$!
sleep 2

cleanup() { kill "$PID" 2>/dev/null || true; }
trap cleanup EXIT

pass() { echo -e "${GREEN}PASS${NC} $1"; }
fail() { echo -e "${RED}FAIL${NC} $1"; exit 1; }

echo "--- Test 1: /health ---"
STATUS=$(curl -sf "$BASE/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
[ "$STATUS" = "ok" ] && pass "/health" || fail "/health returned $STATUS"

echo "--- Test 2: /v1/models ---"
MODEL=$(curl -sf "$BASE/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
[ "$MODEL" = "jimmy/llama3.1-8B" ] && pass "/v1/models" || fail "/v1/models returned $MODEL"

echo "--- Test 3: POST /v1/chat/completions (non-streaming) ---"
RESP=$(curl -sf -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"jimmy/llama3.1-8B","messages":[{"role":"user","content":"Say hello in 3 words"}]}')
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content',''))")
[ -n "$CONTENT" ] && pass "non-streaming content present" || fail "non-streaming empty content"

REASONING=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('reasoning_content','NONE'))")
[ "$REASONING" != "NONE" ] && pass "reasoning_content present" || fail "reasoning_content missing"

echo "--- Test 4: POST /v1/chat/completions (streaming) ---"
STREAM=$(curl -sf -N -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"jimmy/llama3.1-8B","messages":[{"role":"user","content":"Say hello"}],"stream":true}')
echo "$STREAM" | grep -q '\[DONE\]' && pass "streaming [DONE]" || fail "streaming missing [DONE]"
echo "$STREAM" | grep -q 'chat.completion.chunk' && pass "streaming has chunks" || fail "streaming missing chunks"

echo "--- Test 5: Any model accepted ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}')
[ "$CODE" = "200" ] && pass "any model route works" || fail "any model returned $CODE"

echo "--- Test 6: Auth (no API_KEY set) ---"
AUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer wrongkey" \
  -d '{"model":"jimmy/llama3.1-8B","messages":[{"role":"user","content":"hi"}]}')
[ "$AUTH_CODE" != "401" ] && pass "auth disabled, got $AUTH_CODE" || fail "auth returned 401 but no API_KEY"

echo "--- Test 7: Empty messages validation ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"jimmy/llama3.1-8B","messages":[]}')
[ "$CODE" = "400" ] && pass "empty messages returns 400" || fail "empty messages returned $CODE"

echo ""
echo -e "${GREEN}All tests passed!${NC}"
