#!/bin/bash
# Start the backend with fixes applied

echo "🚀 Starting AI Voice Agent Backend with Fixes..."
echo ""
echo "Configuration:"
echo "  📦 Whisper Model: base (faster than medium)"
echo "  ⏱️  Timeouts: 60s Whisper, 120s Ollama"
echo "  🔌 Host: 0.0.0.0:8000"
echo ""

cd "$(dirname "$0")/backend"

echo "Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found, creating..."
    python3 -m venv .venv
fi

echo "✅ Starting backend..."
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
