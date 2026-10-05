import re
import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1
df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

def strip_source_tags(text):
    # removes "CITY (Reuters) - " at the start
    text = re.sub(r"^.{0,100}?\(Reuters\)\s*-?\s*", "", text)
    # removes any other "Reuters" left in the text
    text = re.sub(r"reuters", "", text, flags=re.I)
    return text

df["text"] = df["text"].fillna("").apply(strip_source_tags)