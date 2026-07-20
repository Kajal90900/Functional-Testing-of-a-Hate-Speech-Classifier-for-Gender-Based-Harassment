"""
03_run_model.py

Purpose
-------
Loads the 50-sentence "women" sample prepared by 02_prepare_sample.py,
runs each sentence through the unitary/toxic-bert model, and records the
model's predicted label, confidence score, and whether it matched the
gold (correct) label from HateCheck.

This is a smaller-scale run (50 sentences) as this week's milestone. The
full dataset (several thousand sentences) will be evaluated in a later
stage once this pipeline has been confirmed to work correctly.

Run from the repository root (after running 02_prepare_sample.py first):
    python src/03_run_model.py
"""

import os
import pandas as pd
from transformers import pipeline

INPUT_PATH = os.path.join("data", "hatecheck_women_sample50.csv")
OUTPUT_PATH = os.path.join("results", "model_predictions_week2.csv")
MODEL_NAME = "unitary/toxic-bert"

# toxic-bert returns 6 separate labels; if ANY of these exceed the
# threshold, we treat the sentence as predicted "hateful" overall, to
# compare against HateCheck's single hateful / non-hateful gold label.
TOXIC_LABELS = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}
THRESHOLD = 0.5


def classify_sentence(classifier, sentence):
    predictions = classifier(sentence)[0]  # list of {label, score} for all 6 labels
    scores = {p["label"]: p["score"] for p in predictions}

    # Predicted "hateful" if any toxic-related label crosses the threshold
    is_hateful = any(scores[label] >= THRESHOLD for label in TOXIC_LABELS)
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]

    return {
        "predicted_hateful": is_hateful,
        "top_label": top_label,
        "top_score": round(top_score, 4),
        "toxic_score": round(scores.get("toxic", 0.0), 4),
        "identity_hate_score": round(scores.get("identity_hate", 0.0), 4),
    }


def main():
    print(f"Loading sample from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} sentences.\n")

    print(f"Loading model: {MODEL_NAME} ...")
    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    print("Model loaded successfully.\n")

    results = []
    correct_count = 0

    for i, row in df.iterrows():
        sentence = row["test_case"]
        gold_label = row["label_gold"]  # "hateful" or "non-hateful" in HateCheck
        gold_hateful = (gold_label == "hateful")

        pred = classify_sentence(classifier, sentence)
        is_correct = (pred["predicted_hateful"] == gold_hateful)
        correct_count += int(is_correct)

        results.append({
            "test_case": sentence,
            "functionality": row["functionality"],
            "gold_label": gold_label,
            "predicted_hateful": pred["predicted_hateful"],
            "top_label": pred["top_label"],
            "top_score": pred["top_score"],
            "toxic_score": pred["toxic_score"],
            "identity_hate_score": pred["identity_hate_score"],
            "correct": is_correct,
        })

        print(f"[{i+1:02d}/{len(df)}] gold={gold_label:<12} "
              f"predicted={'hateful' if pred['predicted_hateful'] else 'non-hateful':<12} "
              f"{'CORRECT' if is_correct else 'WRONG'}")

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    accuracy = correct_count / len(df) * 100
    print("\n" + "=" * 50)
    print(f"Accuracy on this 50-sentence sample: {correct_count}/{len(df)} = {accuracy:.1f}%")
    print(f"Full results saved to: {OUTPUT_PATH}")
    print("=" * 50)

    print("\nBreakdown of correctness by functionality category:")
    print(results_df.groupby("functionality")["correct"].mean().sort_values())


if __name__ == "__main__":
    main()
