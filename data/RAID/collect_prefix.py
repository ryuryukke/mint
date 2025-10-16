import pandas as pd

df_raid = pd.read_csv("./test_head.csv")
domains = [
    "abstracts",
    "books",
    "news",
    "poetry",
    "recipes",
    "reddit",
    "reviews",
    "wiki",
]

generators = ["chatgpt", "gpt4", "llama-chat", "gpt2", "mpt-chat"]

for domain in domains:
    for generator in generators:
        df_machine = df_raid[
            (df_raid["domain"] == domain) & (df_raid["model"] == generator)
        ].reset_index(drop=True)
        df_human = df_raid[
            (df_raid["domain"] == domain) & (df_raid["model"] == "human")
        ].reset_index(drop=True)

        positive_sample, negative_sample = (
            df_machine["generation"].to_list()[:10],
            df_human["generation"].to_list()[:10],
        )
        d = {"positive_sample": positive_sample, "negative_sample": negative_sample}
        df_samples = pd.DataFrame(d)
        df_samples.to_json(f"./Prefix/{domain}_{generator}_prefix.json")
