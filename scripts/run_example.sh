#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Evaluating one method (e.g., zlib) on MIAs across one model and one domain
python run.py --task mia --domain arxiv --methods zlib --model_name pythia-160m

# Evaluating one method (e.g., fastdetectgpt) on detection across one model and one domain
python run.py --task detection --domain reddit --methods fastdetectgpt --model_name llama-chat