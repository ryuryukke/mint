#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Evaluating all MIA and detection methods on detection across all models and domains
for MODEL in gpt2 mpt-chat llama-chat chatgpt gpt4; do
    for DOMAIN in abstracts books news poetry recipes reddit reviews wiki; do
        echo "Running for model: $MODEL and domain: $DOMAIN"
        python run.py --task detection --domain $DOMAIN --methods all --model_name $MODEL
    done
done