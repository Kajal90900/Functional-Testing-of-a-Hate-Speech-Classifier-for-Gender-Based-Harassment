# Functional Testing of a Hate Speech Classifier for Gender-Based Harassment

**Capstone Project – Part B**
Student: kajal,meenal,hanan,ali

## Project Overview

This project functionally tests an existing, publicly available hate speech
detection model (`unitary/toxic-bert`) against gender-targeted test cases
drawn from the HateCheck functional test suite (Röttger et al., 2021). Rather
than training a new classifier, the project evaluates how well a
production-grade, pre-trained model identifies gender-based harassment —
including explicit misogyny, implicit/coded misogyny, counter-speech, and
reclaimed language — and reports accuracy, precision, recall, and F1-score.

This work builds directly on the research gaps identified in our Part A
systematic literature review, specifically the **gender-awareness deficit**
and the **explainability gap** in existing harassment detection systems.

## Repository Structure

```
gender-aware-hate-speech-evaluation/
│
├── src/          # Evaluation scripts (to be added from Week 2 onward)
│   └── .gitkeep
│
├── data/         # Dataset files (to be added from Week 2 onward)
│   └── .gitkeep
│
├── results/      # Evaluation outputs — metrics, plots, tables (to be added later)
│   └── .gitkeep
│
├── docs/         # Project logs and report documents
│   └── .gitkeep
│
├── .gitignore
└── README.md
```

## Tools Used

- Git and GitHub (version control), operated through Git Bash
- Python (planned for evaluation scripts — implementation begins Week 2)
- HuggingFace `transformers` and `datasets` libraries (planned)

## Week 1 — Project Setup

This week we focused on setting up the project foundation rather than
writing code:

- Created and organised the GitHub repository into a clear folder structure
  (`src/`, `data/`, `results/`, `docs/`)
- Connected the repository to Git and made multiple tracked commits
- Created a feature branch for our environment planning work and merged it
  into `main` via a pull request, to practise a proper Git workflow
- Wrote this README to document our project title, objective, tools, and
  structure
- Selected our primary methodological reference paper — HateCheck (Röttger
  et al., 2021) — and identified the model we plan to evaluate,
  `unitary/toxic-bert`

No evaluation code has been written yet. Environment setup, dataset loading,
and model loading are planned for Week 2, followed by the full filtering and
evaluation pipeline in subsequent weeks.

## References

Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., & Pierrehumbert,
J. B. (2021). HateCheck: Functional tests for hate speech detection models.
In _Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing_ (pp. 41–58). Association for Computational
Linguistics. https://doi.org/10.18653/v1/2021.acl-long.4

Unitary. (n.d.). _toxic-bert_ (Version 1.0) [Machine learning model]. Hugging
Face. Retrieved July 12, 2026, from https://huggingface.co/unitary/toxic-bert
