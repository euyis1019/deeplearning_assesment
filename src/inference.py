"""
Inference script for Disaster Tweets Classification
Generates submission file for Kaggle

Usage:
    python src/inference.py --model_dir outputs/bertweet_20231207_120000/best_model
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json

from dataset import DisasterTweetDataset


def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """Load trained model and tokenizer"""
    print(f"Loading model from: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    return model, tokenizer


def predict(model, data_loader, device):
    """Generate predictions for the test set"""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of disaster class
    
    return all_preds, all_probs


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load config to determine model type
    config_path = os.path.join(os.path.dirname(args.model_dir), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        is_bertweet = 'bertweet' in config.get('model', '').lower()
        max_length = config.get('max_length', 128)
        print(f"Model type: {config.get('model', 'unknown')}")
    else:
        # Try to infer from model name
        is_bertweet = 'bertweet' in args.model_dir.lower()
        max_length = 128
        print(f"Config not found, inferring model type: {'BERTweet' if is_bertweet else 'BERT'}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)
    
    # Load test data
    print(f"\nLoading test data from: {args.data_dir}")
    test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    test_df['text'] = test_df['text'].fillna('')
    print(f"Test samples: {len(test_df)}")
    
    # Create test dataset and loader
    test_dataset = DisasterTweetDataset(
        texts=test_df['text'].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length,
        is_bertweet=is_bertweet
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions, probabilities = predict(model, test_loader, device)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': predictions
    })
    
    # Save submission
    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, args.output_name)
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ Submission saved to: {submission_path}")
    
    # Print statistics
    print(f"\nPrediction statistics:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted as disaster (1): {sum(predictions)}")
    print(f"  Predicted as not disaster (0): {len(predictions) - sum(predictions)}")
    print(f"  Disaster ratio: {sum(predictions) / len(predictions):.2%}")
    
    # Optionally save probabilities for analysis
    if args.save_probs:
        probs_df = pd.DataFrame({
            'id': test_df['id'],
            'text': test_df['text'],
            'prediction': predictions,
            'disaster_probability': probabilities
        })
        probs_path = os.path.join(args.output_dir, 'predictions_with_probs.csv')
        probs_df.to_csv(probs_path, index=False)
        print(f"  Probabilities saved to: {probs_path}")
    
    return submission_path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions for Disaster Tweets')
    
    # Model
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing test.csv')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./submissions',
                        help='Directory to save submission file')
    parser.add_argument('--output_name', type=str, default='submission.csv',
                        help='Name of the submission file')
    parser.add_argument('--save_probs', action='store_true',
                        help='Save prediction probabilities')
    
    # Other
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

