#!/bin/bash

# Portfolio Build Script
# This script builds the complete portfolio including website and documentation

set -e  # Exit on any error

echo "🚀 Building Mathematical & Computational Sciences Portfolio..."

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "portfolio-website" ]; then
    echo "❌ Error: Please run this script from the repository root"
    exit 1
fi

# Build the portfolio website
echo "📱 Building portfolio website..."
cd portfolio-website

# Check if node_modules exists, install if not
if [ ! -d "node_modules" ]; then
    echo "📦 Installing website dependencies..."
    npm install
fi

# Build the website
echo "🔨 Building website..."
npm run build

echo "✅ Portfolio website built successfully!"
echo "📁 Website files are in: portfolio-website/dist/"

cd ..

# Build Quarto documentation if Quarto is available
if command -v quarto &> /dev/null; then
    echo "📚 Building Quarto documentation..."
    cd quarto-projects
    
    echo "  🔄 Rendering Euler methods analysis..."
    quarto render euler-methods-analysis.qmd
    
    echo "  🔄 Rendering mathematical foundations..."
    quarto render mathematical-foundations.qmd
    
    echo "✅ Quarto documentation built successfully!"
    cd ..
else
    echo "⚠️  Quarto not found - skipping documentation build"
    echo "   Install Quarto from https://quarto.org to build technical documentation"
fi

# Build the C++ project if CMake is available
if command -v cmake &> /dev/null; then
    echo "⚙️  Building C++ numerical methods project..."
    
    # Create build directory if it doesn't exist
    mkdir -p build
    cd build
    
    # Configure and build
    cmake ..
    make -j$(nproc)
    
    echo "✅ C++ project built successfully!"
    cd ..
else
    echo "⚠️  CMake not found - skipping C++ project build"
fi

echo ""
echo "🎉 Portfolio build complete!"
echo ""
echo "📋 Summary:"
echo "  📱 Website: portfolio-website/dist/index.html"
echo "  📚 Documentation: quarto-projects/*.html"
echo "  ⚙️  C++ binaries: build/"
echo ""
echo "🌐 To view the website locally:"
echo "  cd portfolio-website && npm start"
echo ""
echo "📤 Ready for deployment!"