#!/bin/bash
# ============================================================
# RESULTS CLEANUP SCRIPT
# ============================================================
# Removes all results/ directories from vqa/ and preprocessing/
# modules to quickly clean the environment.
#
# Usage:
#   ./clean_results.sh          # Interactive mode (asks for confirmation)
#   ./clean_results.sh -f       # Force mode (no confirmation)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
FORCE=false
if [ "$1" == "-f" ] || [ "$1" == "--force" ]; then
    FORCE=true
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Results Directory Cleanup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Find all results/ directories in vqa/ and preprocessing/
RESULTS_DIRS=()

# Search in vqa/
if [ -d "vqa" ]; then
    while IFS= read -r dir; do
        RESULTS_DIRS+=("$dir")
    done < <(find vqa -type d -name "results" 2>/dev/null)
fi

# Search in preprocessing/
if [ -d "preprocessing" ]; then
    while IFS= read -r dir; do
        RESULTS_DIRS+=("$dir")
    done < <(find preprocessing -type d -name "results" 2>/dev/null)
fi

# Check if any results/ directories were found
if [ ${#RESULTS_DIRS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ No results/ directories found. Environment is already clean.${NC}"
    exit 0
fi

# Display what will be deleted
echo -e "${YELLOW}Found ${#RESULTS_DIRS[@]} results/ director(y/ies):${NC}"
echo ""
for dir in "${RESULTS_DIRS[@]}"; do
    # Calculate directory size
    SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo -e "  ${RED}✗${NC} $dir ${BLUE}($SIZE)${NC}"
done
echo ""

# Ask for confirmation unless in force mode
if [ "$FORCE" = false ]; then
    echo -e "${YELLOW}⚠️  This will permanently delete all files and subdirectories.${NC}"
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Cleanup cancelled.${NC}"
        exit 0
    fi
fi

# Delete directories
echo ""
echo -e "${BLUE}Deleting directories...${NC}"
DELETED_COUNT=0

for dir in "${RESULTS_DIRS[@]}"; do
    if rm -rf "$dir"; then
        echo -e "  ${GREEN}✓${NC} Deleted: $dir"
        ((DELETED_COUNT++))
    else
        echo -e "  ${RED}✗${NC} Failed to delete: $dir"
    fi
done

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo -e "${GREEN}   Deleted: $DELETED_COUNT director(y/ies)${NC}"
echo -e "${BLUE}============================================${NC}"
