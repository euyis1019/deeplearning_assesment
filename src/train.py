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
    get_linear_schedule_with_warmup
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


def train_epoch(model, data_loader, optimizer, scheduler, device):
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
            labels=labels
        )
        
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
            model, train_loader, optimizer, scheduler, device
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

