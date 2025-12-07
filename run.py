#!/usr/bin/env python
"""
Main entry point for Disaster Tweets Classification

Usage:
    # Train with BERTweet (recommended for Twitter data)
    python run.py train --model bertweet --epochs 5
    
    # Train with BERT
    python run.py train --model bert --epochs 5
    
    # Inference
    python run.py predict --model_dir outputs/bertweet_xxx/best_model
    
    # Full pipeline (train + predict)
    python run.py full --model bertweet --epochs 5
"""

import os
import sys
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def train(args):
    """Run training"""
    from src.train import main as train_main, parse_args as train_parse_args
    
    # Build training arguments
    train_args = [
        '--model', args.model,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--seed', str(args.seed),
    ]
    
    # Parse and run
    sys.argv = ['train.py'] + train_args
    parsed_args = train_parse_args()
    return train_main(parsed_args)


def predict(args):
    """Run inference"""
    from src.inference import main as inference_main, parse_args as inference_parse_args
    
    # Build inference arguments
    inference_args = [
        '--model_dir', args.model_dir,
        '--data_dir', args.data_dir,
        '--output_dir', args.submission_dir,
        '--output_name', args.output_name,
        '--batch_size', str(args.batch_size),
    ]
    
    if args.save_probs:
        inference_args.append('--save_probs')
    
    # Parse and run
    sys.argv = ['inference.py'] + inference_args
    parsed_args = inference_parse_args()
    return inference_main(parsed_args)


def full_pipeline(args):
    """Run full pipeline: train + predict"""
    print("="*60)
    print("🚀 Starting Full Pipeline: Train + Predict")
    print("="*60)
    
    # Train
    print("\n📚 Step 1: Training...")
    output_dir = train(args)
    
    # Predict
    print("\n🔮 Step 2: Generating predictions...")
    args.model_dir = os.path.join(output_dir, 'best_model')
    predict(args)
    
    print("\n" + "="*60)
    print("✅ Full pipeline completed!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Disaster Tweets Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train BERTweet model
  python run.py train --model bertweet --epochs 5
  
  # Train BERT model with custom settings
  python run.py train --model bert --epochs 10 --batch_size 16 --learning_rate 3e-5
  
  # Generate predictions
  python run.py predict --model_dir outputs/bertweet_xxx/best_model
  
  # Full pipeline
  python run.py full --model bertweet --epochs 5

Available models:
  - bertweet       : BERTweet base (recommended for Twitter)
  - bertweet-large : BERTweet large
  - bert           : BERT base uncased
  - bert-large     : BERT large uncased
  - roberta        : RoBERTa base
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ============= Train command =============
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, default='bertweet',
                              choices=['bertweet', 'bertweet-large', 'bert', 'bert-large', 'roberta'],
                              help='Model to train')
    train_parser.add_argument('--epochs', type=int, default=5,
                              help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32,
                              help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=2e-5,
                              help='Learning rate')
    train_parser.add_argument('--data_dir', type=str, default='./data',
                              help='Data directory')
    train_parser.add_argument('--output_dir', type=str, default='./outputs',
                              help='Output directory')
    train_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    
    # ============= Predict command =============
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--model_dir', type=str, required=True,
                                help='Path to trained model directory')
    predict_parser.add_argument('--data_dir', type=str, default='./data',
                                help='Data directory')
    predict_parser.add_argument('--submission_dir', type=str, default='./submissions',
                                help='Submission output directory')
    predict_parser.add_argument('--output_name', type=str, default='submission.csv',
                                help='Submission file name')
    predict_parser.add_argument('--batch_size', type=int, default=64,
                                help='Batch size for inference')
    predict_parser.add_argument('--save_probs', action='store_true',
                                help='Save prediction probabilities')
    
    # ============= Full pipeline command =============
    full_parser = subparsers.add_parser('full', help='Run full pipeline (train + predict)')
    full_parser.add_argument('--model', type=str, default='bertweet',
                             choices=['bertweet', 'bertweet-large', 'bert', 'bert-large', 'roberta'],
                             help='Model to train')
    full_parser.add_argument('--epochs', type=int, default=5,
                             help='Number of training epochs')
    full_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    full_parser.add_argument('--learning_rate', type=float, default=2e-5,
                             help='Learning rate')
    full_parser.add_argument('--data_dir', type=str, default='./data',
                             help='Data directory')
    full_parser.add_argument('--output_dir', type=str, default='./outputs',
                             help='Output directory')
    full_parser.add_argument('--submission_dir', type=str, default='./submissions',
                             help='Submission output directory')
    full_parser.add_argument('--output_name', type=str, default='submission.csv',
                             help='Submission file name')
    full_parser.add_argument('--save_probs', action='store_true',
                             help='Save prediction probabilities')
    full_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'full':
        full_pipeline(args)


if __name__ == '__main__':
    main()

