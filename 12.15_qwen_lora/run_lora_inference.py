#!/usr/bin/env python
"""
LoRA Model Inference Script for Disaster Tweets Classification

Supports both single-stage and two-stage LoRA models.

Usage:
    # Single-stage LoRA (attention layers only)
    python run_lora_inference.py \
        --model qwen2.5-1.5b \
        --lora_checkpoint outputs/qwen2.5-1.5b_lora_xxx/best_lora_adapters \
        --output_name submission.csv

    # Two-stage LoRA (attention + MLP layers)
    python run_lora_inference.py \
        --model qwen2.5-1.5b \
        --stage1_checkpoint outputs/qwen2.5-1.5b_lora_xxx/best_lora_adapters \
        --stage2_checkpoint outputs/qwen2.5-1.5b_lora_stage2_xxx/best_stage2_lora_adapters \
        --output_name submission_stage2.csv
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime

# Model configurations (must match training)
LORA_MODEL_CONFIGS = {
    'qwen2.5-1.5b': {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'max_length': 256,
        'prompt_template': '<|im_start|>system\nYou are a disaster detection system. Classify tweets as disaster-related or not.<|im_end|>\n<|im_start|>user\nTweet: {tweet}\nIs this about a real disaster? Answer with "yes" or "no".<|im_end|>\n<|im_start|>assistant\n',
        'yes_token': 'yes',
        'no_token': 'no',
    },
    'phi3.5-mini': {
        'model_name': 'microsoft/Phi-3.5-mini-instruct',
        'max_length': 512,
        'prompt_template': '<|system|>\nYou classify tweets as disaster or not disaster.<|end|>\n<|user|>\n{tweet}\nClassification:<|end|>\n<|assistant|>\n',
        'yes_token': 'disaster',
        'no_token': 'not disaster',
    },
    'llama3.2-1b': {
        'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
        'max_length': 512,
        'prompt_template': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a disaster detection classifier.<|eot_id|><|start_header_id|>user<|end_header_id|>\nTweet: {tweet}\nIs this about a disaster? (yes/no)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n',
        'yes_token': 'yes',
        'no_token': 'no',
    },
    'gemma-2b': {
        'model_name': 'google/gemma-2b-it',
        'max_length': 512,
        'prompt_template': '<start_of_turn>user\nClassify this tweet as disaster or not disaster:\n{tweet}<end_of_turn>\n<start_of_turn>model\n',
        'yes_token': 'disaster',
        'no_token': 'not disaster',
    }
}


def create_prompt(text: str, template: str) -> str:
    """Create prompt from tweet text using template"""
    return template.format(tweet=text)


def load_model(args, config):
    """
    Load LoRA model - supports both single-stage and two-stage

    Returns:
        model: Loaded model with LoRA adapters
        tokenizer: Tokenizer
        yes_token_id: Token ID for positive class
        no_token_id: Token ID for negative class
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print("Loading LoRA Model")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Base model: {config['model_name']}")
    print(f"Device: {device}")

    # Load tokenizer
    print("\nStep 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get token IDs
    yes_token_id = tokenizer.encode(config['yes_token'], add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(config['no_token'], add_special_tokens=False)[0]
    print(f"Yes token: '{config['yes_token']}' (ID: {yes_token_id})")
    print(f"No token: '{config['no_token']}' (ID: {no_token_id})")

    # Load base model
    print(f"\nStep 2: Loading base model: {config['model_name']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
        device_map='auto',
        trust_remote_code=True,
    )

    # Check if single-stage or two-stage
    if args.stage2_checkpoint:
        # Two-stage LoRA
        print("\n🔄 Two-stage LoRA detected")
        print(f"\nStep 3a: Loading Stage 1 LoRA from {args.stage1_checkpoint}")
        stage1_model = PeftModel.from_pretrained(
            base_model,
            args.stage1_checkpoint,
            is_trainable=False
        )

        print("Step 3b: Merging Stage 1 LoRA into base model...")
        merged_model = stage1_model.merge_and_unload()

        print(f"\nStep 3c: Loading Stage 2 LoRA from {args.stage2_checkpoint}")
        model = PeftModel.from_pretrained(
            merged_model,
            args.stage2_checkpoint,
            is_trainable=False
        )
        print("✅ Two-stage LoRA model loaded successfully!")

    else:
        # Single-stage LoRA
        print("\n🔄 Single-stage LoRA detected")
        checkpoint = args.lora_checkpoint or args.stage1_checkpoint
        print(f"\nStep 3: Loading LoRA adapters from {checkpoint}")
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint,
            is_trainable=False
        )
        print("✅ Single-stage LoRA model loaded successfully!")

    model.eval()
    print(f"{'='*60}\n")

    return model, tokenizer, yes_token_id, no_token_id


def predict_batch(model, input_ids, attention_mask, yes_token_id, no_token_id, device):
    """
    Predict for a batch of inputs

    Returns:
        predictions: Binary predictions (0 or 1)
        probabilities: Probabilities for class 1
    """
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get logits for last token
        last_token_logits = outputs.logits[:, -1, :]

        # Extract yes/no token logits
        yes_logits = last_token_logits[:, yes_token_id]
        no_logits = last_token_logits[:, no_token_id]

        # Create binary logits [batch_size, 2]
        binary_logits = torch.stack([no_logits, yes_logits], dim=1)

        # Get predictions and probabilities
        probs = F.softmax(binary_logits, dim=1)
        preds = torch.argmax(binary_logits, dim=1)

        return preds.cpu().numpy(), probs[:, 1].cpu().numpy()


def run_inference(args):
    """Run inference on test set and generate submission file"""

    # Get model config
    if args.model not in LORA_MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(LORA_MODEL_CONFIGS.keys())}")

    config = LORA_MODEL_CONFIGS[args.model]

    # Load data
    print("\nLoading test data...")
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")
    test_df['text'] = test_df['text'].fillna('')
    print(f"Test samples: {len(test_df)}")

    # Load model
    model, tokenizer, yes_token_id, no_token_id = load_model(args, config)
    device = next(model.parameters()).device

    # Prepare prompts
    print("\nPreparing prompts...")
    prompts = [create_prompt(text, config['prompt_template']) for text in test_df['text']]

    # Tokenize
    print("Tokenizing...")
    encodings = tokenizer(
        prompts,
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Create dataset and dataloader
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Run inference
    print(f"\n{'='*60}")
    print("Running Inference")
    print(f"{'='*60}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total batches: {len(dataloader)}")

    all_predictions = []
    all_probabilities = []

    for input_ids, attention_mask in tqdm(dataloader, desc="Predicting"):
        preds, probs = predict_batch(
            model, input_ids, attention_mask,
            yes_token_id, no_token_id, device
        )
        all_predictions.extend(preds)
        all_probabilities.extend(probs)

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': all_predictions
    })

    # Save submission file
    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, args.output_name)
    submission_df.to_csv(submission_path, index=False)

    # Save probabilities if requested
    if args.save_probs:
        probs_path = submission_path.replace('.csv', '_probs.csv')
        probs_df = pd.DataFrame({
            'id': test_df['id'],
            'target': all_predictions,
            'probability': all_probabilities
        })
        probs_df.to_csv(probs_path, index=False)
        print(f"\n✅ Probabilities saved to: {probs_path}")

    # Print statistics
    print(f"\n{'='*60}")
    print("Prediction Statistics")
    print(f"{'='*60}")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Predicted disasters (1): {sum(all_predictions)} ({sum(all_predictions)/len(all_predictions)*100:.2f}%)")
    print(f"Predicted non-disasters (0): {len(all_predictions) - sum(all_predictions)} ({(len(all_predictions) - sum(all_predictions))/len(all_predictions)*100:.2f}%)")
    print(f"\n✅ Submission file saved to: {submission_path}")
    print(f"{'='*60}")

    # Validate submission format
    print("\nValidating submission format...")
    try:
        assert len(submission_df) == len(test_df), "Row count mismatch!"
        assert set(submission_df.columns) == {'id', 'target'}, "Column mismatch!"
        assert submission_df['target'].isin([0, 1]).all(), "Invalid target values!"
        assert submission_df['id'].equals(test_df['id']), "ID mismatch!"
        print("✅ Submission format is valid!")
    except AssertionError as e:
        print(f"❌ Validation error: {e}")

    return submission_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='LoRA Model Inference for Disaster Tweets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-stage LoRA
  python run_lora_inference.py \\
      --model qwen2.5-1.5b \\
      --lora_checkpoint outputs/qwen2.5-1.5b_lora_xxx/best_lora_adapters

  # Two-stage LoRA
  python run_lora_inference.py \\
      --model qwen2.5-1.5b \\
      --stage1_checkpoint outputs/qwen2.5-1.5b_lora_xxx/best_lora_adapters \\
      --stage2_checkpoint outputs/qwen2.5-1.5b_lora_stage2_xxx/best_stage2_lora_adapters
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b',
                        choices=list(LORA_MODEL_CONFIGS.keys()),
                        help='Model type')

    # LoRA checkpoints
    parser.add_argument('--lora_checkpoint', type=str, default='',
                        help='Path to single-stage LoRA checkpoint')
    parser.add_argument('--stage1_checkpoint', type=str, default='',
                        help='Path to Stage 1 LoRA checkpoint (for two-stage)')
    parser.add_argument('--stage2_checkpoint', type=str, default='',
                        help='Path to Stage 2 LoRA checkpoint (for two-stage)')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing test.csv')

    # Output
    parser.add_argument('--output_dir', type=str, default='./submissions',
                        help='Directory to save submission file')
    parser.add_argument('--output_name', type=str, default='submission.csv',
                        help='Submission file name')
    parser.add_argument('--save_probs', action='store_true',
                        help='Save prediction probabilities')

    # Inference settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--use_fp16', action='store_true',
                        help='Use FP16 for inference')

    args = parser.parse_args()

    # Validation
    if not args.stage2_checkpoint:
        # Single-stage mode
        if not args.lora_checkpoint and not args.stage1_checkpoint:
            parser.error("Must provide either --lora_checkpoint or --stage1_checkpoint")
        if not args.lora_checkpoint:
            args.lora_checkpoint = args.stage1_checkpoint
    else:
        # Two-stage mode
        if not args.stage1_checkpoint:
            parser.error("Two-stage mode requires --stage1_checkpoint")

    return args


if __name__ == '__main__':
    args = parse_args()

    print("="*60)
    print("🚀 LoRA Model Inference")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    submission_path = run_inference(args)

    print("\n" + "="*60)
    print("✅ Inference completed!")
    print(f"Submission file: {submission_path}")
    print("="*60)
    print("\nNext steps:")
    print("1. Validate submission format:")
    print(f"   head {submission_path}")
    print("2. Submit to Kaggle:")
    print(f"   kaggle competitions submit -c nlp-getting-started -f {submission_path} -m 'LoRA model submission'")
    print("="*60)
