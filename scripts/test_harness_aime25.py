#!/usr/bin/env python3
"""End-to-end harness mode test: Train a verifier for AIME 2025.

Uses enzu.harness() to autonomously:
1. pip_install scikit-learn and any needed packages
2. Generate training data: AIME-style (response, correct_answer) pairs
3. Train a logistic regression verifier that predicts correct vs wrong
4. Evaluate kimi-k2.5 on real AIME problems
5. Run the trained verifier on model outputs
6. Report results

Run:
    uv run python scripts/test_harness_aime25.py
"""

import enzu


def main() -> None:
    result = enzu.harness(
        goal="""\
You are an autonomous ML harness. Your job is to TRAIN a verifier model
and use it to evaluate LLM outputs on AIME 2025 problems.

== STEP 1: INSTALL ==
Use pip_install("scikit-learn") to install sklearn.
Print confirmation that it worked.

== STEP 2: GENERATE TRAINING DATA ==
Create synthetic training data for a verifier. The verifier should learn
to distinguish correct vs incorrect AIME-style answers.

Generate 200 training examples:
- Each example: (features, label)
- Features: [answer_value, answer_length, has_digits_only, answer_in_range_0_999]
- Label: 1 if answer is correct, 0 if wrong

For "correct" examples: sample answers from known AIME answer distribution (0-999).
For "wrong" examples: use common failure modes (too large, negative, non-numeric
  features like length=0, has_digits_only=0, etc.)

Use random module (already available) to generate this.
Print dataset size and class balance.

== STEP 3: TRAIN VERIFIER ==
Using sklearn (you pip_installed it):
- Split data 80/20 train/test
- Train a LogisticRegression verifier on the training set
- Print test accuracy, precision, recall
- Save the model in memory (just keep the variable)

== STEP 4: EVALUATE LLM ON AIME ==
Define 5 real AIME 2025-I problems:
  P1 (answer=70): Sum of all integer bases b>9 for which 17_b divides 97_b
  P2 (answer=2178): Four-digit abcd with 4*abcd=dcba
  P3 (answer=704): Baseball batting orders, no consecutive all-stars
  P4 (answer=0): Ordered pairs (x,y) positive ints with x^2+y^2=x^3
  P5 (answer=34): Count n<=50 where d(d(n)) is prime

For each problem, use llm_query() with prompt:
  "Solve this AIME problem. Think step by step. On the LAST line,
   write ONLY the integer answer (0-999).\n\n{problem}"

Extract the answer from each response using regex (find last integer).

== STEP 5: RUN TRAINED VERIFIER ==
For each LLM response:
- Extract features: [answer_value, len(answer_str), is_digits_only, in_range]
- Run through trained LogisticRegression
- Compare verifier prediction vs ground truth

Print a table:
  Problem | Ground Truth | LLM Answer | Verifier Says | Actually Correct

== STEP 6: REPORT ==
Print:
- LLM accuracy: X/5
- Verifier accuracy (did it correctly predict pass/fail?): X/5
- Verifier model test set accuracy from Step 3
- Conclusion: does the verifier agree with ground truth?

Call FINAL() with the complete report string.
""",
        budget={"tokens": 200_000, "minutes": 15},
        model="moonshotai/kimi-k2.5",
        max_steps=30,
    )

    print("\n" + "=" * 60)
    print("HARNESS RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
