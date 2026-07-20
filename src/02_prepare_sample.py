"""
02_prepare_sample.py

Purpose
-------
Loads the HateCheck functional test suite (Rottger et al., 2021), filters
it to the "women" target-identity subset, and randomly samples exactly 50
sentences for this week's evaluation. The full women-only subset (likely
several hundred sentences) will be used later; this week's scope is
intentionally limited to a 50-sentence sample.

A fixed random seed is used so the same 50 sentences are selected every
time this script is run, making the sample reproducible.

Run from the repository root:
    python src/02_prepare_sample.py
"""

import os
import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = os.path.join("data", "hatecheck_women_sample50.csv")
TARGET_GROUP = "women"
SAMPLE_SIZE = 50
RANDOM_SEED = 42


def main():
    print("Loading HateCheck dataset from HuggingFace...")
    dataset = load_dataset("Paul/hatecheck", split="test")
    df = dataset.to_pandas()
    print(f"Full dataset loaded: {len(df)} total test cases\n")

    # Filter to the gender-targeted ("women") subset only
    women_df = df[df["target_ident"] == TARGET_GROUP].reset_index(drop=True)
    print(f"Filtered to target group = '{TARGET_GROUP}': {len(women_df)} test cases")

    # Take a fixed, reproducible random sample of 50 sentences
    sample_df = women_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nSampled {len(sample_df)} sentences for this week's evaluation.")
    print("\nBreakdown by gold label (hateful vs non-hateful):")
    print(sample_df["label_gold"].value_counts())

    print("\nBreakdown by functionality category:")
    print(sample_df["functionality"].value_counts())

    os.makedirs("data", exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved 50-sentence sample to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
