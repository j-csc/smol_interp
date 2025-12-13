#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up smol_interp development environment...${NC}"

# Check for required tools
echo -e "${BLUE}Checking for required tools...${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v bun &> /dev/null; then
    echo -e "${RED}Error: bun is not installed${NC}"
    echo "Install it with: curl -fsSL https://bun.sh/install | bash"
    exit 1
fi

# Setup backend
echo -e "${GREEN}Setting up backend...${NC}"
cd backend
uv sync
echo -e "${GREEN}Backend dependencies installed!${NC}"

# Setup frontend
echo -e "${GREEN}Setting up frontend...${NC}"
cd ../frontend
bun install
echo -e "${GREEN}Frontend dependencies installed!${NC}"

echo -e "${BLUE}Setup complete! Run ./run.sh to start the development environment.${NC}"
