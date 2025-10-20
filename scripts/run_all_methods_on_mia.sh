#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Evaluating all MIA and detection methods on MIAs across all models and domains
for MODEL in pythia-160m pythia-1.4b pythia-2.8b pythia-6.9b pythia-12b; do
    for DOMAIN in wikipedia_\(en\) github pile_cc pubmed_central arxiv dm_mathematics hackernews; do
        echo "Running for model: $MODEL and domain: $DOMAIN"
        python run.py --task mia --domain $DOMAIN --methods all --model_name $MODEL
    done
done