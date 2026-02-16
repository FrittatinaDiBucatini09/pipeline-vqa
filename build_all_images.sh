#!/bin/bash
# build_all_images.sh
# Recursively finds and executes all 'build_image*.sh' scripts in the project.

set -e

echo "🔍 Searching for build scripts..."

# Find all scripts matching the pattern, excluding hidden directories
# Using process substitution to allow the loop to run in the main shell context if needed (though subshell is fine here)
while IFS= read -r script; do
    echo ""
    echo "=============================================================================="
    echo "🚀 FOUND BUILD SCRIPT: $script"
    echo "=============================================================================="
    
    DIR=$(dirname "$script")
    SCRIPT_NAME=$(basename "$script")
    
    # Check if the script is executable, make it executable if not
    if [ ! -x "$script" ]; then
        echo "⚠️  Script is not executable. Fixing permissions..."
        chmod +x "$script"
    fi
    
    # Execute in a subshell to isolate directory changes
    (
        echo "Vg  Entering directory: $DIR"
        cd "$DIR"
        
        echo "▶️  Executing ./$SCRIPT_NAME..."
        ./"$SCRIPT_NAME"
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ SUCCESS: $script completed successfully."
        else
            echo "❌ FAILURE: $script failed with exit code $EXIT_CODE."
            exit $EXIT_CODE
        fi
    )
    
    # Catch failure from subshell if set -e is active
    if [ $? -ne 0 ]; then
        echo "⛔ Aborting global build due to failure in $script"
        exit 1
    fi
    
done < <(find . -type f -name "build_image*.sh" -not -path "*/.*" | sort)

echo ""
echo "🎉 ALL BUILDS COMPLETED SUCCESSFULLY!"
