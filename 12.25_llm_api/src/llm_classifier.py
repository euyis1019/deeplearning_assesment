#!/usr/bin/env python3
"""
LLM-based Few-Shot Learning Classifier for Disaster Tweets
Uses external LLM API with carefully designed prompts and few-shot examples
"""

import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time
import random


class LLMDisasterClassifier:
    """
    Disaster tweet classifier using LLM API with few-shot learning
    """

    def __init__(
        self,
        api_url: str = "https://newapi.deepwisdom.ai/v1/chat/completions",
        api_key: str = "sk-DUuJeAxX6fwNViL2JHRIHj4SI9OEcDlr4TMctin2DLxvuFY8",
        model: str = "deepseek-v3",
        temperature: float = 0.1,
        max_tokens: int = 10,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize LLM classifier

        Args:
            api_url: API endpoint URL
            api_key: API authentication key
            model: Model name to use
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def create_few_shot_prompt(self, text: str, examples: List[Dict]) -> List[Dict]:
        """
        Create few-shot learning prompt with examples

        Args:
            text: Tweet text to classify
            examples: List of example dicts with 'text' and 'label' keys

        Returns:
            List of messages for the API
        """
        # System prompt defining the task
        system_prompt = """You are a disaster tweet classifier. Your task is to determine if a tweet is about a REAL disaster or not.

A tweet is labeled as 1 (disaster) if it describes:
- Natural disasters (earthquakes, floods, hurricanes, wildfires, etc.)
- Accidents (plane crashes, building collapses, explosions, etc.)
- Emergency situations (fires, evacuations, etc.)
- Deaths or casualties from disasters
- Urgent warnings about imminent danger

A tweet is labeled as 0 (not disaster) if it:
- Uses disaster words metaphorically (e.g., "my room is a disaster")
- Discusses movies, games, or fiction about disasters
- Contains general news without urgent disaster context
- Uses hyperbole or exaggeration
- Discusses past disasters in a non-urgent context

Analyze the tweet carefully and respond with ONLY the number 0 or 1."""

        # Create few-shot examples
        few_shot_messages = [{"role": "system", "content": system_prompt}]

        # Add examples
        for example in examples:
            few_shot_messages.append({
                "role": "user",
                "content": f"Tweet: {example['text']}"
            })
            few_shot_messages.append({
                "role": "assistant",
                "content": str(example['label'])
            })

        # Add the actual query
        few_shot_messages.append({
            "role": "user",
            "content": f"Tweet: {text}"
        })

        return few_shot_messages

    def call_api(self, messages: List[Dict]) -> Optional[str]:
        """
        Call LLM API with retry logic

        Args:
            messages: List of message dicts for the conversation

        Returns:
            Response text or None if failed
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                        return content.strip()
                else:
                    print(f"API error (attempt {attempt + 1}/{self.max_retries}): Status {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

            except requests.exceptions.Timeout:
                print(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def parse_response(self, response: str) -> int:
        """
        Parse LLM response to get binary prediction

        Args:
            response: Raw response from LLM

        Returns:
            0 or 1 prediction
        """
        if response is None:
            return 0  # Default to non-disaster if API fails

        # Try to extract 0 or 1 from response
        response = response.strip()

        # Direct match
        if response == "0":
            return 0
        elif response == "1":
            return 1

        # Look for 0 or 1 in response
        if "1" in response and "0" not in response:
            return 1
        elif "0" in response and "1" not in response:
            return 0

        # Check for keywords as fallback
        response_lower = response.lower()
        if any(word in response_lower for word in ["yes", "disaster", "real", "true"]):
            return 1
        elif any(word in response_lower for word in ["no", "not", "false", "metaphor"]):
            return 0

        # Default to 0 if unclear
        print(f"Warning: Unclear response '{response}', defaulting to 0")
        return 0

    def predict_single(self, text: str, examples: List[Dict]) -> int:
        """
        Predict single tweet

        Args:
            text: Tweet text
            examples: Few-shot examples

        Returns:
            0 or 1 prediction
        """
        messages = self.create_few_shot_prompt(text, examples)
        response = self.call_api(messages)
        prediction = self.parse_response(response)
        return prediction

    def predict_batch(
        self,
        texts: List[str],
        examples: List[Dict],
        batch_delay: float = 0.5
    ) -> List[int]:
        """
        Predict batch of tweets

        Args:
            texts: List of tweet texts
            examples: Few-shot examples
            batch_delay: Delay between API calls to avoid rate limiting

        Returns:
            List of predictions
        """
        predictions = []

        for text in tqdm(texts, desc="Classifying tweets"):
            pred = self.predict_single(text, examples)
            predictions.append(pred)

            # Add delay to avoid rate limiting
            if batch_delay > 0:
                time.sleep(batch_delay)

        return predictions


def select_few_shot_examples(
    train_df: pd.DataFrame,
    n_examples: int = 8,
    balanced: bool = True,
    random_seed: int = 42
) -> List[Dict]:
    """
    Select few-shot examples from training data

    Args:
        train_df: Training dataframe with 'text' and 'target' columns
        n_examples: Number of examples to select
        balanced: Whether to balance examples across classes
        random_seed: Random seed for reproducibility

    Returns:
        List of example dicts
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    examples = []

    if balanced:
        # Select equal number from each class
        n_per_class = n_examples // 2

        # Get examples for each class
        class_0 = train_df[train_df['target'] == 0].sample(n=n_per_class, random_state=random_seed)
        class_1 = train_df[train_df['target'] == 1].sample(n=n_per_class, random_state=random_seed)

        # Combine and shuffle
        selected = pd.concat([class_0, class_1]).sample(frac=1, random_state=random_seed)
    else:
        # Random selection
        selected = train_df.sample(n=n_examples, random_state=random_seed)

    # Convert to example dicts
    for _, row in selected.iterrows():
        examples.append({
            'text': row['text'],
            'label': row['target']
        })

    return examples


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Evaluate predictions

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics
