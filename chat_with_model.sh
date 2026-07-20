#!/bin/bash

# Check if a checkpoint path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint-path>"
  echo "Example: $0 ./models/lfm2-350m/checkpoint-step-20"
  exit 1
fi

CHECKPOINT_PATH=$1

# Run the chat script
python -m liquid_foundation_model.chat --model-path "$CHECKPOINT_PATH"