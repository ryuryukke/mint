#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# load the pile dataset
python collect_pile.py

# collect prefixes for a specific domain
for DOMAIN in wikipedia_\(en\) github pile_cc pubmed_central arxiv dm_mathematics hackernews; do
python collect_prefix.py --domain $DOMAIN --sample_size 10
done