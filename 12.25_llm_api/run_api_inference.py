#!/usr/bin/env python3
"""
Run LLM API-based inference with few-shot learning for disaster tweets classification
"""

import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_classifier import LLMDisasterClassifier, select_few_shot_examples, evaluate_predictions


def load_data(data_dir: str = "data"):
    """Load train and test data"""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(test_df)} test samples")

    return train_df, test_df


def run_inference(
    n_examples: int = 8,
    batch_delay: float = 0.5,
    temperature: float = 0.1,
    evaluate_on_val: bool = False,
    val_size: int = 100,
    output_dir: str = "../submissions",
    api_url: str = "https://newapi.deepwisdom.ai/v1/chat/completions",
    api_key: str = "sk-DUuJeAxX6fwNViL2JHRIHj4SI9OEcDlr4TMctin2DLxvuFY8",
    model: str = "deepseek-v3"
):
    """
    Run inference using LLM API with few-shot learning

    Args:
        n_examples: Number of few-shot examples to use
        batch_delay: Delay between API calls (seconds)
        temperature: LLM temperature
        evaluate_on_val: Whether to evaluate on validation set first
        val_size: Size of validation set
        output_dir: Directory to save submissions
        api_url: API endpoint URL
        api_key: API key
        model: Model name
    """
    print("=" * 80)
    print("LLM API-based Few-Shot Learning for Disaster Tweets")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()

    # Initialize classifier
    print(f"\nInitializing LLM classifier...")
    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Few-shot examples: {n_examples}")

    classifier = LLMDisasterClassifier(
        api_url=api_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=10,
        timeout=30
    )

    # Select few-shot examples
    print(f"\nSelecting {n_examples} few-shot examples from training data...")
    examples = select_few_shot_examples(
        train_df,
        n_examples=n_examples,
        balanced=True,
        random_seed=42
    )

    print("\nFew-shot examples:")
    for i, ex in enumerate(examples, 1):
        label_str = "DISASTER" if ex['label'] == 1 else "NOT DISASTER"
        print(f"  {i}. [{label_str}] {ex['text'][:80]}...")

    # Optional: Evaluate on validation set
    if evaluate_on_val:
        print(f"\n{'=' * 80}")
        print(f"Evaluating on {val_size} validation samples...")
        print("=" * 80)

        val_df = train_df.sample(n=val_size, random_state=42)
        val_texts = val_df['text'].tolist()
        val_labels = val_df['target'].tolist()

        val_predictions = classifier.predict_batch(
            val_texts,
            examples,
            batch_delay=batch_delay
        )

        # Evaluate
        metrics = evaluate_predictions(val_labels, val_predictions)
        print("\nValidation Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

    # Run on test set
    print(f"\n{'=' * 80}")
    print(f"Running inference on {len(test_df)} test samples...")
    print("=" * 80)

    test_texts = test_df['text'].tolist()
    test_predictions = classifier.predict_batch(
        test_texts,
        examples,
        batch_delay=batch_delay
    )

    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_predictions
    })

    # Save submission
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_fewshot_{model.replace('/', '_')}_{n_examples}shot_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    submission_df.to_csv(filepath, index=False)
    print(f"\nSubmission saved to: {filepath}")

    # Show statistics
    print("\nPrediction statistics:")
    print(f"  Total predictions: {len(test_predictions)}")
    print(f"  Disaster (1):      {sum(test_predictions)} ({sum(test_predictions)/len(test_predictions)*100:.1f}%)")
    print(f"  Not disaster (0):  {len(test_predictions) - sum(test_predictions)} ({(len(test_predictions) - sum(test_predictions))/len(test_predictions)*100:.1f}%)")

    # Save config
    config = {
        'model': model,
        'temperature': temperature,
        'n_examples': n_examples,
        'batch_delay': batch_delay,
        'timestamp': timestamp,
        'examples': examples
    }

    config_file = filepath.replace('.csv', '_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_file}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM API-based inference with few-shot learning"
    )

    parser.add_argument(
        '--n_examples',
        type=int,
        default=8,
        help='Number of few-shot examples (default: 8)'
    )

    parser.add_argument(
        '--batch_delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature (default: 0.1)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate on validation set before test set'
    )

    parser.add_argument(
        '--val_size',
        type=int,
        default=100,
        help='Validation set size (default: 100)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../submissions',
        help='Output directory for submissions (default: ../submissions)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-v3',
        help='LLM model name (default: deepseek-v3)'
    )

    parser.add_argument(
        '--api_url',
        type=str,
        default='https://newapi.deepwisdom.ai/v1/chat/completions',
        help='API endpoint URL'
    )

    parser.add_argument(
        '--api_key',
        type=str,
        default='sk-[your-api-key]',
        help='API key'
    )

    args = parser.parse_args()

    run_inference(
        n_examples=args.n_examples,
        batch_delay=args.batch_delay,
        temperature=args.temperature,
        evaluate_on_val=args.evaluate,
        val_size=args.val_size,
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model
    )


if __name__ == "__main__":
    main()
