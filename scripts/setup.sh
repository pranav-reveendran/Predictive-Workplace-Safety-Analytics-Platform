#!/bin/bash

# Predictive Workplace Safety Analytics Platform
# Setup Script

set -e  # Exit on any error

echo "🏗️  Setting up Predictive Workplace Safety Analytics Platform..."
echo "================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p logs
mkdir -p exports
mkdir -p reports

# Copy configuration template
echo "⚙️  Setting up configuration..."
if [ ! -f config/config.yaml ]; then
    cp config/config.template.yaml config/config.yaml
    echo "✅ Configuration file created. Please edit config/config.yaml with your settings."
else
    echo "⚠️  Configuration file already exists."
fi

# Check PostgreSQL
echo "🗄️  Checking PostgreSQL availability..."
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL client found"
else
    echo "⚠️  PostgreSQL client not found. Please install PostgreSQL."
fi

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Set up pre-commit hooks
echo "🎣 Setting up pre-commit hooks..."
pre-commit install

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Edit config/config.yaml with your database settings"
echo "   3. Initialize the database: python src/database/init_db.py"
echo "   4. Download OSHA data: python src/data/download_osha_data.py"
echo "   5. Process the data: python src/data/process_data.py"
echo "   6. Train the model: python src/models/train_model.py"
echo ""
echo "🐳 For Docker deployment:"
echo "   docker-compose up --build"
echo ""
echo "📊 For Tableau dashboard:"
echo "   Connect Tableau to your PostgreSQL database using the configured settings"
echo ""
echo "Happy analyzing! 🔍" 