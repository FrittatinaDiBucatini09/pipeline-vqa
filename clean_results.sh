#!/bin/bash
# ============================================================
# RESULTS CLEANUP SCRIPT
# ============================================================
# Removes all results/ directories from vqa/ and preprocessing/
# AND the orchestrator_runs/ directory.
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
echo -e "${BLUE}Project Cleanup Utility${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Find directories to clean
DIRS_TO_CLEAN=()

# 1. Orchestrator Runs (PRESERVED)
# if [ -d "orchestrator_runs" ]; then
#     DIRS_TO_CLEAN+=("orchestrator_runs")
# fi


# 2. Search in vqa/ for results
if [ -d "vqa" ]; then
    while IFS= read -r dir; do
        DIRS_TO_CLEAN+=("$dir")
    done < <(find vqa -type d -name "results" 2>/dev/null)
fi

# 3. Search in preprocessing/ for results
if [ -d "preprocessing" ]; then
    while IFS= read -r dir; do
        DIRS_TO_CLEAN+=("$dir")
    done < <(find preprocessing -type d -name "results" 2>/dev/null)
fi

# Check if anything was found
if [ ${#DIRS_TO_CLEAN[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ No directories to clean found.${NC}"
else
    # Display what will be deleted
    echo -e "${YELLOW}Found ${#DIRS_TO_CLEAN[@]} directory(ies) to remove:${NC}"
    echo ""
    for dir in "${DIRS_TO_CLEAN[@]}"; do
        # Calculate directory size
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo -e "  ${RED}✗${NC} $dir ${BLUE}($SIZE)${NC}"
    done
    echo ""

    # Ask for confirmation unless in force mode
    if [ "$FORCE" = false ]; then
        echo -e "${YELLOW}⚠️  This will permanently delete these directories and all their contents.${NC}"
        read -p "Do you want to proceed? (y/N): " -n 1 -r
        echo ""

        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Directory cleanup cancelled.${NC}"
            # Don't exit, continue to logs
        else
             # Delete directories
            echo ""
            echo -e "${BLUE}Deleting directories...${NC}"
            
            for dir in "${DIRS_TO_CLEAN[@]}"; do
                if rm -rf "$dir" 2>/dev/null; then
                    echo -e "  ${GREEN}✓${NC} Deleted: $dir"
                    ((++DELETED_COUNT))
                else
                    echo -e "  ${YELLOW}⚠${NC} Standard delete failed. Trying cleanup via Docker..."
                    # Resolve absolute path for Docker (bind mounts require absolute paths)
                    if [ -d "$dir" ]; then
                        ABS_PARENT_DIR="$(cd "$(dirname "$dir")" && pwd)"
                        TARGET_NAME="$(basename "$dir")"
                        
                        if docker run --rm -v "${ABS_PARENT_DIR}":/clean_target -w /clean_target alpine rm -rf "${TARGET_NAME}"; then
                             echo -e "  ${GREEN}✓${NC} Deleted (via Docker): $dir"
                             ((++DELETED_COUNT))
                        else
                             echo -e "  ${RED}✗${NC} Failed to delete: $dir"
                        fi
                    else
                        echo -e "  ${YELLOW}⚠${NC} Directory already gone: $dir"
                    fi
                fi
            done
        fi
    else
         # Force mode - just delete
        echo ""
        echo -e "${BLUE}Deleting directories...${NC}"
        for dir in "${DIRS_TO_CLEAN[@]}"; do
            rm -rf "$dir" 2>/dev/null || true # Try standard
             # If still exists, try docker (lazy check)
            if [ -d "$dir" ]; then
                 ABS_PARENT_DIR="$(cd "$(dirname "$dir")" && pwd)"
                 TARGET_NAME="$(basename "$dir")"
                 docker run --rm -v "${ABS_PARENT_DIR}":/clean_target -w /clean_target alpine rm -rf "${TARGET_NAME}"
            fi
            echo -e "  ${GREEN}✓${NC} Processed: $dir"
            ((++DELETED_COUNT))
        done
    fi
fi

# ============================================================
# LOG FILES CLEANUP (*.err, *.out)
# ============================================================

echo ""
echo -e "${BLUE}Scanning for log files (*.err, *.out)...${NC}"

# Find all .err and .out files (excluding hidden directories like .git)
# Using process substitution to avoid subshell variable scope issues
LOG_FILES=()
while IFS= read -r file; do
    LOG_FILES+=("$file")
done < <(find . -type f \( -name "*.err" -o -name "*.out" \) -not -path "*/.*/*")

LOG_COUNT=${#LOG_FILES[@]}

if [ "$LOG_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Found $LOG_COUNT log file(s).${NC}"
    
    # Confirmation for logs (reuse FORCE flag or ask if not forced)
    PROCEED_LOGS=$FORCE
    if [ "$FORCE" = false ]; then
        echo ""
        read -p "Do you want to delete these log files too? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            PROCEED_LOGS=true
        else
            echo -e "${BLUE}Log cleanup skipped.${NC}"
        fi
    fi

    if [ "$PROCEED_LOGS" = true ]; then
        echo -e "${BLUE}Deleting log files...${NC}"
        for log in "${LOG_FILES[@]}"; do
            rm -f "$log"
            echo -e "  ${GREEN}✓${NC} Deleted: $log"
        done
        echo -e "${GREEN}✅ Deleted $LOG_COUNT log file(s).${NC}"
    fi
else
    echo -e "${GREEN}✅ No log files found.${NC}"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo -e "${GREEN}   Deleted: $DELETED_COUNT directory(ies)${NC}"
echo -e "${BLUE}============================================${NC}"
