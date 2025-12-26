"""
Dataset classes for Disaster Tweets Classification
Supports both BERTweet and BERT models
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re
import html


class DisasterTweetDataset(Dataset):
    """
    PyTorch Dataset for Disaster Tweets
    
    Args:
        texts: List of tweet texts
        labels: List of labels (optional, None for test set)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        is_bertweet: Whether using BERTweet (applies special preprocessing)
    """
    
    def __init__(
        self,
        texts: list,
        labels: list = None,
        tokenizer: AutoTokenizer = None,
        max_length: int = 128,
        is_bertweet: bool = True
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_bertweet = is_bertweet
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Preprocess text
        text = self.preprocess_tweet(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item
    
    def preprocess_tweet(self, text: str) -> str:
        """
        Enhanced tweet preprocessing with better text normalization

        For BERTweet: Uses normalization similar to training data
        For BERT: Standard cleaning
        """
        # Decode HTML entities
        text = html.unescape(text)

        # Normalize repeated characters (e.g., "sooooo" -> "soo", "!!!" -> "!!")
        # Keep at most 2 repetitions to preserve emphasis
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Split camelCase hashtags (e.g., #DisasterRelief -> disaster relief)
        # This helps the model understand compound hashtags
        def split_hashtag(match):
            hashtag = match.group(1)
            # Split on uppercase letters
            words = re.sub(r'([A-Z][a-z]+)', r' \1', hashtag).strip()
            return words.lower() if words else hashtag.lower()

        text = re.sub(r'#(\w+)', split_hashtag, text)

        if self.is_bertweet:
            # BERTweet expects: @USER for mentions, HTTPURL for URLs
            # Replace user mentions
            text = re.sub(r'@\w+', '@USER', text)
            # Replace URLs
            text = re.sub(r'http\S+|www\S+|https\S+', 'HTTPURL', text, flags=re.MULTILINE)
        else:
            # For standard BERT, clean more aggressively
            # Replace user mentions
            text = re.sub(r'@\w+', '', text)
            # Replace URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()


def load_data(data_dir: str):
    """
    Load train and test data from CSV files
    
    Args:
        data_dir: Directory containing train.csv and test.csv
        
    Returns:
        train_df, test_df: Pandas DataFrames
    """
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    
    # Fill NaN values in text column (if any)
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Loaded {len(test_df)} test samples")
    print(f"Label distribution in training set:")
    print(train_df['target'].value_counts())
    
    return train_df, test_df


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    is_bertweet: bool = True,
    num_workers: int = 4
):
    """
    Create PyTorch DataLoaders for train, validation and test sets
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    train_dataset = DisasterTweetDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['target'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        is_bertweet=is_bertweet
    )
    
    val_dataset = DisasterTweetDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['target'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        is_bertweet=is_bertweet
    )
    
    test_dataset = DisasterTweetDataset(
        texts=test_df['text'].tolist(),
        labels=None,  # No labels for test set
        tokenizer=tokenizer,
        max_length=max_length,
        is_bertweet=is_bertweet
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoTokenizer
    
    # Load data
    train_df, test_df = load_data("./data")
    
    # Test with BERTweet tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    
    dataset = DisasterTweetDataset(
        texts=train_df['text'].tolist()[:5],
        labels=train_df['target'].tolist()[:5],
        tokenizer=tokenizer,
        is_bertweet=True
    )
    
    print("\nSample data:")
    for i in range(3):
        item = dataset[i]
        print(f"\nOriginal: {train_df['text'].iloc[i]}")
        print(f"Preprocessed: {dataset.preprocess_tweet(train_df['text'].iloc[i])}")
        print(f"Input IDs shape: {item['input_ids'].shape}")
        print(f"Label: {item['labels']}")

