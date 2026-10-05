import pandas as pd

CORPUS_FILE = "clean-100k.csv"   # change to the real CSV name from the unzipped file
MAX_PER_DOMAIN = 3000            # stops one website from dominating the data

# 1. Kaggle Fake and Real News (our convention: fake = 0, real = 1)
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1
kaggle = pd.concat([fake, true], ignore_index=True)[["title", "text", "label"]]
kaggle["source"] = "kaggle"

# 2. WELFake (its labels are flipped, so we fix them)
wel = pd.read_csv("WELFake_Dataset.csv")[["title", "text", "label"]]
wel["label"] = 1 - wel["label"]
wel["source"] = "welfake"

# 3. FakeNewsCorpus subset (labels are per website, so they are a bit noisy)
corp = pd.read_csv(CORPUS_FILE, low_memory=False)
corp = corp[corp["type"].isin(["reliable", "fake", "conspiracy"])].copy()
corp["label"] = (corp["type"] == "reliable").astype(int)   # reliable = 1, others = 0
corp = corp.rename(columns={"content": "text"})

# cap rows per website, so no single site dominates
corp = corp.sample(frac=1, random_state=42)
corp = corp.groupby("domain").head(MAX_PER_DOMAIN)
print("Corpus rows kept:", len(corp))
print("Domains per class:\n", corp.groupby("label")["domain"].nunique())

corp = corp[["title", "text", "label"]]
corp["source"] = "corpus"

# 4. Merge (earlier sources first, so their copies are kept when deduplicating)
df = pd.concat([kaggle, wel, corp], ignore_index=True)
print("\nAfter merge:", len(df))

# 5. Remove rows with no text
df["text"] = df["text"].fillna("")
df["title"] = df["title"].fillna("")
df = df[df["text"].str.strip() != ""]
print("After removing empty text:", len(df))

# 6. Remove duplicates (compare simplified versions of title and text)
df["title_key"] = df["title"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
df["text_key"] = (df["text"].str.lower()
                  .str.replace(r"[^a-z0-9]", "", regex=True)
                  .str[:300])

has_title = df["title_key"] != ""
df = pd.concat([df[has_title].drop_duplicates("title_key"), df[~has_title]])
df = df.drop_duplicates("text_key")
print("After removing duplicates:", len(df))

# 7. Safety check: same article should not appear with both labels
conflicts = df.groupby("text_key")["label"].nunique()
print("Articles with conflicting labels:", (conflicts > 1).sum())

# 8. Save
df = df[["title", "text", "label", "source"]].sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("merged_raw.csv", index=False)

print("\nFinal size:", len(df))
print(df["label"].value_counts())
print(df["source"].value_counts())