"""
Training script for Disaster Tweets Classification
Supports BERTweet and BERT models

Usage:
    python src/train.py --model bertweet --epochs 5 --batch_size 32
    python src/train.py --model bert --epochs 5 --batch_size 32
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import json
from datetime import datetime

from dataset import DisasterTweetDataset, load_data, create_data_loaders


# Model configurations
MODEL_CONFIGS = {
    'bertweet': {
        'model_name': 'vinai/bertweet-base',
        'is_bertweet': True,
        'max_length': 128,
    },
    'bertweet-large': {
        'model_name': 'vinai/bertweet-large',
        'is_bertweet': True,
        'max_length': 128,
    },
    'bert': {
        'model_name': 'bert-base-uncased',
        'is_bertweet': False,
        'max_length': 128,
    },
    'bert-large': {
        'model_name': 'bert-large-uncased',
        'is_bertweet': False,
        'max_length': 128,
    },
    'roberta': {
        'model_name': 'roberta-base',
        'is_bertweet': False,
        'max_length': 128,
    }
}


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like F1, 'min' for metrics like loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch(model, data_loader, optimizer, scheduler, device, use_label_smoothing=False, label_smoothing=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if not use_label_smoothing else None
        )

        # Apply label smoothing if enabled
        if use_label_smoothing:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            loss = loss_fct(outputs.logits, labels)
        else:
            loss = outputs.loss

        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='binary')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc


def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='binary')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc, all_preds, all_labels


def main(args):
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get model config
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[args.model]
    print(f"\nModel: {args.model}")
    print(f"Model name: {config['model_name']}")
    
    # Load data
    print("\n" + "="*50)
    print("Loading data...")
    train_df, test_df = load_data(args.data_dir)
    
    # Split train into train and validation
    train_data, val_data = train_test_split(
        train_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=train_df['target']
    )
    print(f"\nTrain size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    
    # Load tokenizer and model
    print("\n" + "="*50)
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Configure dropout if custom values are provided
    if args.attention_dropout > 0 or args.hidden_dropout > 0 or args.classifier_dropout > 0:
        print(f"Using custom dropout configuration:")
        print(f"  Attention dropout: {args.attention_dropout}")
        print(f"  Hidden dropout: {args.hidden_dropout}")
        print(f"  Classifier dropout: {args.classifier_dropout}")

        model_config = AutoConfig.from_pretrained(
            config['model_name'],
            num_labels=2,
            attention_probs_dropout_prob=args.attention_dropout if args.attention_dropout > 0 else None,
            hidden_dropout_prob=args.hidden_dropout if args.hidden_dropout > 0 else None,
            classifier_dropout=args.classifier_dropout if args.classifier_dropout > 0 else None
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            config=model_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=2
        )

    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("\n" + "="*50)
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data,
        val_data,
        test_df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=config['max_length'],
        is_bertweet=config['is_bertweet'],
        num_workers=args.num_workers
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if args.use_cosine_schedule:
        print("Using cosine annealing scheduler")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )
    else:
        print("Using linear scheduler")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Training
    print("\n" + "="*50)
    print("Starting training...")

    # Initialize early stopping if enabled
    early_stopping = None
    if args.use_early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode='max'  # For F1 score
        )
        print(f"Early stopping enabled (patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})")

    best_f1 = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': []
    }

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*50)

        # Train
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_label_smoothing=args.use_label_smoothing,
            label_smoothing=args.label_smoothing
        )
        
        # Evaluate
        val_loss, val_f1, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, device
        )
        
        # Log metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            print(f"\n🎉 New best F1 score! Saving model...")

            # Save model
            model.save_pretrained(os.path.join(output_dir, 'best_model'))
            tokenizer.save_pretrained(os.path.join(output_dir, 'best_model'))

            # Save classification report
            report = classification_report(val_labels, val_preds, target_names=['Not Disaster', 'Disaster'])
            print(f"\nClassification Report:\n{report}")
            with open(os.path.join(output_dir, 'best_classification_report.txt'), 'w') as f:
                f.write(report)

        # Check early stopping
        if early_stopping is not None:
            if early_stopping(val_f1):
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
                break
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training config
    training_config = {
        'model': args.model,
        'model_name': config['model_name'],
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': config['max_length'],
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'seed': args.seed
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best F1 score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {output_dir}/best_model")
    print("="*50)
    
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train Disaster Tweets Classifier')
    
    # Model
    parser.add_argument('--model', type=str, default='bertweet',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model to use for training')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing train.csv and test.csv')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate scheduler')

    # Regularization
    parser.add_argument('--use_label_smoothing', action='store_true',
                        help='Use label smoothing')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--attention_dropout', type=float, default=0.0,
                        help='Attention dropout probability (0 to use model default)')
    parser.add_argument('--hidden_dropout', type=float, default=0.0,
                        help='Hidden layer dropout probability (0 to use model default)')
    parser.add_argument('--classifier_dropout', type=float, default=0.0,
                        help='Classifier dropout probability (0 to use model default)')

    # Scheduler
    parser.add_argument('--use_cosine_schedule', action='store_true',
                        help='Use cosine annealing scheduler instead of linear')

    # Early Stopping
    parser.add_argument('--use_early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum improvement required for early stopping')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save model and results')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

