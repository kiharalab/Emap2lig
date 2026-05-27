#!/usr/bin/env bash
# Emap2lig quick setup script
# Usage: bash scripts/setup.sh
# Verifies prerequisites and installs the project.

set -euo pipefail

echo "🔍 Checking prerequisites..."

# Check uv
if ! command -v uv &>/dev/null; then
	echo "❌ uv not found. Install from https://docs.astral.sh/uv/getting-started/installation/"
	echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
	exit 1
fi
echo "✅ uv found: $(uv --version)"

# Check Python (3.12+)
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if [ "$(echo "$PY_VERSION" | cut -d. -f1)" -lt 3 ] || { [ "$(echo "$PY_VERSION" | cut -d. -f1)" -eq 3 ] && [ "$(echo "$PY_VERSION" | cut -d. -f2)" -lt 12 ]; }; then
	echo "❌ Python 3.12+ required, found: $PY_VERSION"
	echo "   Install via uv: uv python install 3.12"
	exit 1
fi
echo "✅ Python $PY_VERSION"

# Check CUDA GPU (optional)
if command -v nvidia-smi &>/dev/null; then
	echo "✅ CUDA GPU detected:"
	nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "   (query failed)"
else
	echo "⚠️  No nvidia-smi found — CPU-only mode will be slow"
fi

# Install project
echo ""
echo "📦 Installing Emap2lig..."
cd "$(dirname "$0")/../../.."
uv sync
echo "✅ Installation complete"

# Check Node.js for web GUI (optional)
if command -v node &>/dev/null; then
	echo "✅ Node.js found: $(node --version)"
	echo "   To install web GUI: uv sync --group web"
else
	echo "⚠️  Node.js not found — web GUI requires Node.js 18+"
fi

echo ""
echo "🚀 Ready! Try:"
echo "   uv run emap2lig --help"
echo "   uv run fragment-detect --help"
echo "   uv run --group web python app/start.py  (web GUI)"
