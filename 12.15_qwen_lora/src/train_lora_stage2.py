"""
Progressive LoRA Training - Stage 2
Loads Stage 1 LoRA (attention layers), merges it, then adds new LoRA on MLP layers

This script:
1. Loads base model (Qwen2.5-1.5B)
2. Loads and merges Stage 1 LoRA adapters (attention layers)
3. Adds new LoRA on MLP layers (gate_proj, up_proj, down_proj)
4. Trains only the new LoRA parameters

Usage:
    python src/train_lora_stage2.py \
        --model qwen2.5-1.5b \
        --stage1_checkpoint ./outputs/qwen2.5-1.5b_lora_20251215_174427/best_lora_adapters \
        --epochs 5 \
        --batch_size 16 \
        --lora_r 24
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import json
from datetime import datetime

# Model configurations for 1B-2B models
LORA_MODEL_CONFIGS = {
    'qwen2.5-1.5b': {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'max_length': 256,
        'prompt_template': '<|im_start|>system\nYou are a disaster detection system. Classify tweets as disaster-related or not.<|im_end|>\n<|im_start|>user\nTweet: {tweet}\nIs this about a real disaster? Answer with "yes" or "no".<|im_end|>\n<|im_start|>assistant\n',
        'yes_token': 'yes',
        'no_token': 'no',
        # Stage 2 targets MLP layers
        'stage2_target_modules': ['gate_proj', 'up_proj', 'down_proj'],
    },
    'phi3.5-mini': {
        'model_name': 'microsoft/Phi-3.5-mini-instruct',
        'max_length': 512,
        'prompt_template': '<|system|>\nYou classify tweets as disaster or not disaster.<|end|>\n<|user|>\n{tweet}\nClassification:<|end|>\n<|assistant|>\n',
        'yes_token': 'disaster',
        'no_token': 'not disaster',
        'stage2_target_modules': ['gate_proj', 'up_proj', 'down_proj'],
    },
    'llama3.2-1b': {
        'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
        'max_length': 512,
        'prompt_template': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a disaster detection classifier.<|eot_id|><|start_header_id|>user<|end_header_id|>\nTweet: {tweet}\nIs this about a disaster? (yes/no)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n',
        'yes_token': 'yes',
        'no_token': 'no',
        'stage2_target_modules': ['gate_proj', 'up_proj', 'down_proj'],
    },
    'gemma-2b': {
        'model_name': 'google/gemma-2b-it',
        'max_length': 512,
        'prompt_template': '<start_of_turn>user\nClassify this tweet as disaster or not disaster:\n{tweet}<end_of_turn>\n<start_of_turn>model\n',
        'yes_token': 'disaster',
        'no_token': 'not disaster',
        'stage2_target_modules': ['gate_proj', 'up_proj', 'down_proj'],
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


def load_data(data_dir: str):
    """Load train and test data"""
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')

    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(test_df)} test samples")
    print(f"Label distribution:")
    print(train_df['target'].value_counts())

    return train_df, test_df


def create_prompt(text: str, template: str) -> str:
    """Create prompt from tweet text using template"""
    return template.format(tweet=text)


def prepare_data_for_lora(df, tokenizer, config, max_length, is_train=True):
    """Prepare data with prompts for LoRA training"""
    prompts = []
    labels = []

    for idx, row in df.iterrows():
        prompt = create_prompt(row['text'], config['prompt_template'])
        prompts.append(prompt)
        if is_train:
            labels.append(row['target'])

    # Tokenize prompts
    encodings = tokenizer(
        prompts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
    }

    if is_train:
        dataset_dict['labels'] = torch.tensor(labels, dtype=torch.long)

    return dataset_dict


class LoRADataset(torch.utils.data.Dataset):
    """PyTorch Dataset for LoRA training"""
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item


def get_stage2_lora_config(args, target_modules):
    """Create Stage 2 LoRA configuration for MLP layers"""
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    return config


def compute_classification_loss(logits, labels, yes_token_id, no_token_id):
    """
    Compute classification loss by comparing probabilities of yes/no tokens

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: True labels [batch_size]
        yes_token_id: Token ID for "yes" (disaster)
        no_token_id: Token ID for "no" (not disaster)
    """
    # Get logits for the last token (where the answer should be)
    last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

    # Extract probabilities for yes/no tokens
    yes_logits = last_token_logits[:, yes_token_id]  # [batch_size]
    no_logits = last_token_logits[:, no_token_id]   # [batch_size]

    # Stack to create binary classification logits
    binary_logits = torch.stack([no_logits, yes_logits], dim=1)  # [batch_size, 2]

    # Compute cross-entropy loss
    loss = F.cross_entropy(binary_logits, labels)

    # Get predictions
    preds = torch.argmax(binary_logits, dim=1)

    return loss, preds


def train_epoch(model, data_loader, optimizer, scheduler, device, yes_token_id, no_token_id):
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

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Compute classification loss
        loss, preds = compute_classification_loss(
            outputs.logits, labels, yes_token_id, no_token_id
        )

        total_loss += loss.item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='binary')
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, acc


def evaluate(model, data_loader, device, yes_token_id, no_token_id):
    """Evaluate model"""
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
            )

            loss, preds = compute_classification_loss(
                outputs.logits, labels, yes_token_id, no_token_id
            )

            total_loss += loss.item()
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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Get model config
    if args.model not in LORA_MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(LORA_MODEL_CONFIGS.keys())}")

    config = LORA_MODEL_CONFIGS[args.model]
    print(f"\n{'='*60}")
    print(f"Progressive LoRA Training - Stage 2")
    print(f"Model: {args.model}")
    print(f"Model name: {config['model_name']}")
    print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data(args.data_dir)

    # Split train/val (or use full training data)
    if args.val_ratio > 0:
        train_data, val_data = train_test_split(
            train_df,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=train_df['target']
        )
        print(f"\nTrain size: {len(train_data)}")
        print(f"Validation size: {len(val_data)}")
    else:
        # Use all training data, no validation
        train_data = train_df
        val_data = None
        print(f"\nUsing full training data (no validation split): {len(train_data)} samples")
        print("WARNING: No validation set - cannot monitor overfitting!")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get token IDs for yes/no
    yes_token_id = tokenizer.encode(config['yes_token'], add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(config['no_token'], add_special_tokens=False)[0]
    print(f"\nYes token: '{config['yes_token']}' (ID: {yes_token_id})")
    print(f"No token: '{config['no_token']}' (ID: {no_token_id})")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_encodings = prepare_data_for_lora(
        train_data, tokenizer, config, config['max_length'], is_train=True
    )
    train_dataset = LoRADataset(train_encodings)

    # Create data loaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Create validation loader only if validation data exists
    if val_data is not None:
        val_encodings = prepare_data_for_lora(
            val_data, tokenizer, config, config['max_length'], is_train=True
        )
        val_dataset = LoRADataset(val_encodings)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    else:
        val_loader = None

    # Step 1: Load base model
    print(f"\n{'='*60}")
    print("STAGE 2 - PROGRESSIVE LORA TRAINING")
    print(f"{'='*60}")
    print(f"\nStep 1/3: Loading base model: {config['model_name']}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
        device_map='auto',
        trust_remote_code=True,
    )

    # Step 2: Load and merge Stage 1 LoRA
    print(f"\nStep 2/3: Loading Stage 1 LoRA from {args.stage1_checkpoint}")
    print("This contains trained attention layer adapters...")

    stage1_model = PeftModel.from_pretrained(
        base_model,
        args.stage1_checkpoint,
        is_trainable=False  # We don't need to train these
    )

    print("\nMerging Stage 1 LoRA into base model...")
    print("(This combines the trained attention layers with the base model)")
    merged_model = stage1_model.merge_and_unload()

    # Step 3: Add Stage 2 LoRA on MLP layers
    print(f"\nStep 3/3: Adding Stage 2 LoRA on MLP layers...")
    print(f"Target modules: {config['stage2_target_modules']}")
    print(f"LoRA rank (r): {args.lora_r} (increased from Stage 1)")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"LoRA dropout: {args.lora_dropout}")

    stage2_lora_config = get_stage2_lora_config(args, config['stage2_target_modules'])
    model = get_peft_model(merged_model, stage2_lora_config)

    print("\n" + "="*60)
    print("Model Parameter Summary:")
    print("="*60)
    model.print_trainable_parameters()
    print("\nNote: Only Stage 2 LoRA (MLP layers) will be trained.")
    print("Stage 1 LoRA (attention layers) has been merged and frozen.")
    print("="*60)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )

    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model}_lora_stage2_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Training
    print("\n" + "="*60)
    print("Starting Stage 2 LoRA training...")
    print("="*60)

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
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)

        # Train
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            yes_token_id, no_token_id
        )

        # Log metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_acc'].append(train_acc)

        # Evaluate only if validation set exists
        if val_loader is not None:
            val_loss, val_f1, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, device,
                yes_token_id, no_token_id
            )
            print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")

            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            history['val_acc'].append(val_acc)

            # Save best model based on validation F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                print(f"\nNew best F1 score! Saving Stage 2 LoRA adapters...")

                # Save only Stage 2 LoRA adapters
                model.save_pretrained(os.path.join(output_dir, 'best_stage2_lora_adapters'))
                tokenizer.save_pretrained(os.path.join(output_dir, 'best_stage2_lora_adapters'))

                # Save classification report
                report = classification_report(
                    val_labels, val_preds,
                    target_names=['Not Disaster', 'Disaster']
                )
                print(f"\nClassification Report:\n{report}")
                with open(os.path.join(output_dir, 'best_classification_report.txt'), 'w') as f:
                    f.write(report)
        else:
            # No validation - save model after each epoch
            print(f"\nSaving checkpoint for epoch {epoch + 1}...")
            model.save_pretrained(os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}'))
            tokenizer.save_pretrained(os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}'))

            # Always update best based on training (since no validation)
            best_f1 = train_f1
            best_epoch = epoch + 1

            # Also save as "best" (will be overwritten each epoch)
            model.save_pretrained(os.path.join(output_dir, 'best_stage2_lora_adapters'))
            tokenizer.save_pretrained(os.path.join(output_dir, 'best_stage2_lora_adapters'))

    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    training_config = {
        'model': args.model,
        'model_name': config['model_name'],
        'stage1_checkpoint': args.stage1_checkpoint,
        'stage2_target_modules': config['stage2_target_modules'],
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'seed': args.seed
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(training_config, f, indent=2)

    print("\n" + "="*60)
    print("Stage 2 LoRA Training completed!")
    print(f"Best F1 score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Stage 2 LoRA adapters saved to: {output_dir}/best_stage2_lora_adapters")
    print("\nNote: To use this model for inference, you need to:")
    print("1. Load base model")
    print(f"2. Load Stage 1 LoRA and merge: {args.stage1_checkpoint}")
    print(f"3. Load Stage 2 LoRA: {output_dir}/best_stage2_lora_adapters")
    print("="*60)

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Progressive LoRA Training - Stage 2')

    # Model
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b',
                        choices=list(LORA_MODEL_CONFIGS.keys()),
                        help='Model to use for LoRA training')

    # Stage 1 checkpoint (REQUIRED)
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to Stage 1 LoRA checkpoint (attention layers)')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing train.csv and test.csv')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')

    # Stage 2 LoRA Configuration (MLP layers)
    parser.add_argument('--lora_r', type=int, default=24,
                        help='LoRA rank for Stage 2 (default 24, higher than Stage 1)')
    parser.add_argument('--lora_alpha', type=int, default=48,
                        help='LoRA alpha for Stage 2 (default 48)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')

    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--use_fp16', action='store_true',
                        help='Use FP16 mixed precision')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save model and results')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
