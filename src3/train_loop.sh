#!/bin/bash

# Script to run training multiple times in a loop to free memory between runs
# Usage: ./train_loop.sh <num_iterations> [additional_args]
# Example: ./train_loop.sh 5 --episodes 1000 --headless

set -e  # Exit on error

# Check if number of iterations is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <num_iterations> [additional_args]"
    echo "Example: $0 5 --episodes 1000 --headless"
    echo "Example: $0 10 --episodes 500 --headless --save_every_n 100"
    exit 1
fi

NUM_ITERATIONS=$1
shift  # Remove first argument, rest are additional args

# Base command
BASE_CMD="python train.py g1.yaml --train"

# Additional arguments (if any)
ADDITIONAL_ARGS="$@"

# Default episodes if not specified
if [[ ! "$ADDITIONAL_ARGS" =~ --episodes ]]; then
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --episodes 1000"
fi

# Default headless if not specified
if [[ ! "$ADDITIONAL_ARGS" =~ --headless ]]; then
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --headless"
fi

echo "=========================================="
echo "Training Loop Script"
echo "=========================================="
echo "Number of iterations: $NUM_ITERATIONS"
echo "Base command: $BASE_CMD"
echo "Additional args: $ADDITIONAL_ARGS"
echo "=========================================="
echo ""

# Run training iterations
for i in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo "=========================================="
    echo "Iteration $i / $NUM_ITERATIONS"
    echo "=========================================="
    echo "Starting at: $(date)"
    
    # First iteration: start fresh (no --load_pretrained)
    # Subsequent iterations: load previous model (with --load_pretrained)
    if [ $i -eq 1 ]; then
        CMD="$BASE_CMD $ADDITIONAL_ARGS"
        echo "Command: $CMD (fresh start)"
    else
        CMD="$BASE_CMD $ADDITIONAL_ARGS --load_pretrained"
        echo "Command: $CMD (loading previous model)"
    fi
    
    # Run the training command
    if $CMD; then
        echo ""
        echo "✓ Iteration $i completed successfully at $(date)"
    else
        EXIT_CODE=$?
        echo ""
        echo "✗ Iteration $i failed with exit code $EXIT_CODE at $(date)"
        echo "Stopping training loop."
        exit $EXIT_CODE
    fi
    
    echo "Waiting 2 seconds before next iteration..."
    sleep 2
done

echo ""
echo "=========================================="
echo "All $NUM_ITERATIONS iterations completed!"
echo "=========================================="
echo "Finished at: $(date)"

