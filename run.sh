#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting smol_interp development environment...${NC}"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill 0
}
trap cleanup EXIT

# Start Modal backend server
echo -e "${GREEN}Starting Modal backend server...${NC}"
cd backend
source .venv/bin/activate
modal serve app.py &
BACKEND_PID=$!

# Wait a bit for Modal to initialize
sleep 3

# Start frontend dev server
echo -e "${GREEN}Starting frontend dev server...${NC}"
cd ../frontend
bun run dev &
FRONTEND_PID=$!

# Wait for both processes
wait
