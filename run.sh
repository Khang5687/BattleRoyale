#!/bin/bash

# BattleRoyale5 - Build and Run Script
# This script builds the Vulkan C++ battle royale simulation and runs it

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 BattleRoyale5 - Build and Run Script${NC}"
echo "============================================"

# Get script directory (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo -e "${YELLOW}📁 Project root: $PROJECT_ROOT${NC}"

# Check for required tools
echo -e "${BLUE}🔧 Checking dependencies...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}❌ Error: cmake not found. Please install CMake.${NC}"
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo -e "${RED}❌ Error: make not found. Please install build tools.${NC}"
    exit 1
fi

if ! pkg-config --exists glfw3; then
    echo -e "${RED}❌ Error: GLFW3 not found. Please install via Homebrew: brew install glfw${NC}"
    exit 1
fi

if ! pkg-config --exists vulkan; then
    echo -e "${YELLOW}⚠️  Warning: Vulkan pkg-config not found. Trying to find Vulkan SDK...${NC}"
    if [ -z "$VULKAN_SDK" ]; then
        echo -e "${RED}❌ Error: VULKAN_SDK environment variable not set. Please install Vulkan SDK.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Dependencies check passed${NC}"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}📂 Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Check if we need to run cmake configure
NEED_CONFIGURE=false
if [ ! -f "CMakeCache.txt" ]; then
    NEED_CONFIGURE=true
    echo -e "${YELLOW}🔧 No CMakeCache.txt found, will configure...${NC}"
elif [ "$PROJECT_ROOT/CMakeLists.txt" -nt "CMakeCache.txt" ]; then
    NEED_CONFIGURE=true
    echo -e "${YELLOW}🔧 CMakeLists.txt newer than cache, will reconfigure...${NC}"
fi

# Configure with CMake if needed
if [ "$NEED_CONFIGURE" = true ]; then
    echo -e "${BLUE}⚙️  Configuring project with CMake...${NC}"
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo "$PROJECT_ROOT"
    echo -e "${GREEN}✅ CMake configuration complete${NC}"
fi

# Build the project
echo -e "${BLUE}🔨 Building project...${NC}"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ ! -f "$BUILD_DIR/battleroyale5" ]; then
    echo -e "${RED}❌ Error: Build failed, executable not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build complete${NC}"

# Copy assets and configuration files if they've changed
echo -e "${BLUE}📦 Copying assets...${NC}"

# Copy assets directory
if [ -d "$PROJECT_ROOT/assets" ]; then
    rsync -a --delete "$PROJECT_ROOT/assets/" "$BUILD_DIR/assets/"
    echo -e "${GREEN}✅ Assets copied${NC}"
else
    echo -e "${YELLOW}⚠️  No assets directory found${NC}"
fi

# Copy bias.txt if it exists
if [ -f "$PROJECT_ROOT/bias.txt" ]; then
    cp "$PROJECT_ROOT/bias.txt" "$BUILD_DIR/"
    echo -e "${GREEN}✅ bias.txt copied${NC}"
fi

# Check if shaders were compiled
if [ -f "$BUILD_DIR/shaders/circle.vert.spv" ] && [ -f "$BUILD_DIR/shaders/circle.frag.spv" ]; then
    echo -e "${GREEN}✅ Shaders compiled successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: Shader compilation may have failed${NC}"
fi

echo "============================================"
echo -e "${GREEN}🎮 Starting BattleRoyale5...${NC}"
echo "============================================"

# Run the executable from the project root so relative paths work correctly
cd "$PROJECT_ROOT"
exec "$BUILD_DIR/battleroyale5"
