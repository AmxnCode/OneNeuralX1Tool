#!/bin/bash

# Install required packages
pip install transformers datasets tqdm

# Train the small model
python -m liquid_foundation_model.training.train \
  --model-size 350M \
  --dataset wikitext \
  --output-dir ./models/lfm2-350m \
  --tokenizer-path "LiquidAI/LFM2-350M" \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --max-steps 100 \
  --warmup-steps 10 \
  --num-epochs 3 \
  --max-length 512

# Test text generation
python -m liquid_foundation_model.inference \
  --model-path ./models/lfm2-350m/final \
  --prompt "Once upon a time, there was a" \
  --max-length 100 \
  --temperature 0.3 \
  --min-p 0.15 \
  --repetition-penalty 1.05