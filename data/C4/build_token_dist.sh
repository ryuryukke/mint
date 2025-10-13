# retrieve C4 data (the file num 15 is a default of the original DC-PDD paper)
for i in $(seq -w 0 14); do
  fname="c4-train.000${i}-of-01024.json.gz"
  wget "https://huggingface.co/datasets/allenai/c4/resolve/main/en/${fname}"
done
# compute token frequency distribution via a model tokenizer
python comp_token_dist.py