#!/bin/bash

# BattleRoyale5 Config Tool - Run Script
# This script runs the damage curve configuration tool

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}⚙️  BattleRoyale5 - Damage Curve Config Tool${NC}"
echo "============================================"

# Get script directory (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
CONFIG_EXECUTABLE="$BUILD_DIR/battleroyale5-config"

echo -e "${YELLOW}📁 Project root: $PROJECT_ROOT${NC}"

# Check if we need to build the config tool
NEED_BUILD=false

if [ ! -f "$CONFIG_EXECUTABLE" ]; then
    NEED_BUILD=true
    echo -e "${YELLOW}🔧 Config executable not found, will build...${NC}"
elif [ "$PROJECT_ROOT/src/config_main.cpp" -nt "$CONFIG_EXECUTABLE" ]; then
    NEED_BUILD=true
    echo -e "${YELLOW}🔧 Source code newer than executable, will rebuild...${NC}"
elif [ "$PROJECT_ROOT/src/damage_curve.cpp" -nt "$CONFIG_EXECUTABLE" ]; then
    NEED_BUILD=true
    echo -e "${YELLOW}🔧 Damage curve source newer than executable, will rebuild...${NC}"
elif [ "$PROJECT_ROOT/CMakeLists.txt" -nt "$CONFIG_EXECUTABLE" ]; then
    NEED_BUILD=true
    echo -e "${YELLOW}🔧 CMakeLists.txt newer than executable, will rebuild...${NC}"
fi

# Build if needed
if [ "$NEED_BUILD" = true ]; then
    echo -e "${BLUE}🔨 Building config tool...${NC}"

    # Create build directory if it doesn't exist
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}📂 Creating build directory...${NC}"
        mkdir -p "$BUILD_DIR"
    fi

    cd "$BUILD_DIR"

    # Configure with CMake if needed
    if [ ! -f "CMakeCache.txt" ]; then
        echo -e "${BLUE}⚙️  Configuring project with CMake...${NC}"
        cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo "$PROJECT_ROOT"
    fi

    # Build just the config tool
    make battleroyale5-config -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    # Check if build was successful
    if [ ! -f "$CONFIG_EXECUTABLE" ]; then
        echo -e "${RED}❌ Error: Config tool build failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Config tool build complete${NC}"
else
    echo -e "${GREEN}✅ Config tool is up to date${NC}"
fi

echo "============================================"
echo -e "${GREEN}🎛️  Starting Damage Curve Editor...${NC}"
echo "============================================"

# Run the config tool from the project root so relative paths work correctly
cd "$PROJECT_ROOT"
exec "$CONFIG_EXECUTABLE"