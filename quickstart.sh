#!/bin/bash
# Quick start script for Text-to-SQL system

set -e

echo "🚀 Text-to-SQL System - Quick Start"
echo "===================================="

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version detected"

# Create virtual environment
echo ""
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Download dataset (optional)
echo ""
read -p "📊 Download Spider dataset? (y/n): " download_data
if [ "$download_data" = "y" ]; then
    echo "Downloading Spider dataset..."
    python src/data_preparation.py --download
    echo "✅ Dataset downloaded"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/{spider,databases,preprocessed}
mkdir -p models/{t5_finetuned,tokenizer}
mkdir -p notebooks
echo "✅ Directories created"

# Quick test
echo ""
echo "🧪 Running quick test..."
python -c "import torch; import transformers; print('✅ Core libraries working')"

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train model:    python train.py --model_name t5-base --epochs 10"
echo "2. Evaluate:       python evaluate.py --model_path ./models/t5_finetuned --test_data ./data/spider/dev.json"
echo "3. Launch app:     streamlit run app.py"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"
echo "============================================"
