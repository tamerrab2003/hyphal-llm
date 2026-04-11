#!/bin/bash
# Resilient download with auto-resume for flaky connections
URL="https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true"
OUT="models/Qwen3-1.7B-Q4_K_M.gguf"
EXPECTED_SIZE=1282439264

MAX_RETRIES=200
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    CURRENT_SIZE=0
    if [ -f "$OUT" ]; then
        CURRENT_SIZE=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
    fi
    
    if [ "$CURRENT_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "Download complete! File size: $CURRENT_SIZE bytes"
        exit 0
    fi
    
    PCT=$(echo "scale=1; $CURRENT_SIZE * 100 / $EXPECTED_SIZE" | bc 2>/dev/null || echo "?")
    echo "Attempt $((RETRY+1)): resuming from $CURRENT_SIZE bytes ($PCT%)..."
    curl -L -C - -o "$OUT" --connect-timeout 10 --max-time 120 --retry 3 "$URL" 2>&1
    RETRY=$((RETRY+1))
    sleep 1
done

echo "Failed after $MAX_RETRIES retries"
exit 1
