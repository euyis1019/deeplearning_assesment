"""
Prediction Correction Script
Corrects Stage 2 predictions based on training data patterns and heuristics
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

def extract_disaster_keywords(train_df):
    """Extract common disaster-related keywords from training data"""
    disaster_texts = train_df[train_df['target'] == 1]['text'].values
    non_disaster_texts = train_df[train_df['target'] == 0]['text'].values

    # Common disaster keywords (based on the dataset)
    disaster_keywords = [
        'fire', 'flood', 'earthquake', 'disaster', 'emergency', 'evacuation',
        'destroyed', 'damage', 'injured', 'death', 'killed', 'dead', 'victim',
        'crash', 'collision', 'explosion', 'bombing', 'terrorist', 'attack',
        'hurricane', 'tornado', 'tsunami', 'wildfire', 'storm', 'drought',
        'rescue', 'survivor', 'casualties', 'threat', 'danger', 'warning',
        'collapse', 'wreckage', 'flames', 'burning', 'blaze', 'inferno',
        'aftershock', 'tremor', 'eruption', 'devastation', 'catastrophe',
        'arson', 'accident', 'derailment', 'sinking', 'drowning'
    ]

    # Metaphorical/non-literal usage keywords (often not real disasters)
    metaphor_indicators = [
        'body', 'hair', 'love', 'game', 'screen', 'fashion', 'new', 'video',
        'song', 'album', 'show', 'movie', 'news', 'tonight', 'like', 'just',
        'get', 'got', 'going', 'via'
    ]

    return disaster_keywords, metaphor_indicators


def has_url(text):
    """Check if text contains URL"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return bool(re.search(url_pattern, text))


def count_disaster_keywords(text, keywords):
    """Count disaster keywords in text"""
    text_lower = text.lower()
    count = sum(1 for kw in keywords if kw in text_lower)
    return count


def analyze_text_features(text, disaster_keywords, metaphor_indicators):
    """Analyze text features for disaster likelihood"""
    features = {
        'disaster_keyword_count': count_disaster_keywords(text, disaster_keywords),
        'metaphor_indicator_count': count_disaster_keywords(text, metaphor_indicators),
        'has_url': has_url(text),
        'length': len(text),
        'word_count': len(text.split()),
        'has_exclamation': '!' in text,
        'has_question': '?' in text,
        'has_hashtag': '#' in text,
    }
    return features


def load_data():
    """Load training, test data and predictions"""
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    submission_df = pd.read_csv('./submissions/submission_stage2.csv')

    # Load probabilities if available
    try:
        probs_df = pd.read_csv('./submissions/submission_stage2_probs.csv')
        # Merge and handle probability column
        submission_df = submission_df.merge(probs_df[['id', 'probability']], on='id', how='left', suffixes=('', '_prob'))

        # Convert single probability to prob_0 and prob_1
        # probability is the probability of the predicted class
        submission_df['prob_1'] = submission_df.apply(
            lambda x: x['probability'] if x['target'] == 1 else 1 - x['probability'], axis=1
        )
        submission_df['prob_0'] = 1 - submission_df['prob_1']
    except Exception as e:
        print(f"Warning: Could not load probability file: {e}")
        submission_df['prob_0'] = 0.5
        submission_df['prob_1'] = 0.5

    return train_df, test_df, submission_df


def correct_predictions(train_df, test_df, submission_df, max_corrections=40):
    """
    Correct predictions based on heuristic rules

    Rules:
    1. Strong disaster keywords + predicted 0 + low confidence -> change to 1
    2. Many metaphor indicators + predicted 1 + low confidence -> change to 0
    3. Very short text (< 10 chars) often not real disasters
    4. URL presence pattern analysis
    """

    # Extract keywords
    disaster_keywords, metaphor_indicators = extract_disaster_keywords(train_df)

    # Merge test data with predictions
    merged_df = test_df.merge(submission_df, on='id', how='left')

    corrections = []

    for idx, row in merged_df.iterrows():
        text = str(row['text'])
        pred = row['target']
        prob_0 = row.get('prob_0', 0.5)
        prob_1 = row.get('prob_1', 0.5)

        # Calculate confidence (distance from 0.5)
        confidence = abs(prob_1 - 0.5)

        # Extract features
        features = analyze_text_features(text, disaster_keywords, metaphor_indicators)

        should_correct = False
        new_pred = pred
        reason = ""

        # Rule 1: Strong disaster signal but predicted 0 with low confidence
        if (pred == 0 and
            features['disaster_keyword_count'] >= 2 and
            confidence < 0.25):
            should_correct = True
            new_pred = 1
            reason = f"Multiple disaster keywords ({features['disaster_keyword_count']}), low confidence"

        # Rule 2: Metaphorical language but predicted 1 with low confidence
        elif (pred == 1 and
              features['metaphor_indicator_count'] >= 3 and
              features['disaster_keyword_count'] == 0 and
              confidence < 0.20):
            should_correct = True
            new_pred = 0
            reason = f"Metaphorical indicators ({features['metaphor_indicator_count']}), no disaster keywords"

        # Rule 3: Very short non-informative text predicted as disaster
        elif (pred == 1 and
              features['word_count'] < 3 and
              confidence < 0.25):
            should_correct = True
            new_pred = 0
            reason = f"Very short text ({features['word_count']} words)"

        # Rule 4: Strong disaster keyword with URL but predicted 0
        elif (pred == 0 and
              features['disaster_keyword_count'] >= 1 and
              features['has_url'] and
              confidence < 0.20):
            should_correct = True
            new_pred = 1
            reason = f"Disaster keywords + URL, likely news"

        if should_correct:
            corrections.append({
                'id': row['id'],
                'text': text[:100],
                'old_pred': pred,
                'new_pred': new_pred,
                'prob_0': prob_0,
                'prob_1': prob_1,
                'confidence': confidence,
                'reason': reason,
                **features
            })

    # Sort by confidence (correct least confident predictions first)
    corrections_df = pd.DataFrame(corrections)
    if len(corrections_df) > 0:
        corrections_df = corrections_df.sort_values('confidence').head(max_corrections)

    return corrections_df


def apply_corrections(submission_df, corrections_df):
    """Apply corrections to submission"""
    corrected_submission = submission_df.copy()

    for _, correction in corrections_df.iterrows():
        idx = corrected_submission[corrected_submission['id'] == correction['id']].index
        if len(idx) > 0:
            corrected_submission.loc[idx, 'target'] = correction['new_pred']

    return corrected_submission[['id', 'target']]


def main():
    print("Loading data...")
    train_df, test_df, submission_df = load_data()

    print(f"\nOriginal submission stats:")
    print(submission_df['target'].value_counts())

    print("\nAnalyzing patterns and finding corrections...")
    corrections_df = correct_predictions(train_df, test_df, submission_df, max_corrections=40)

    print(f"\nFound {len(corrections_df)} potential corrections")

    if len(corrections_df) > 0:
        print("\nTop corrections to apply:")
        print("="*120)
        for idx, row in corrections_df.head(20).iterrows():
            print(f"ID {row['id']}: {row['old_pred']} -> {row['new_pred']} "
                  f"(conf: {row['confidence']:.3f}, prob_1: {row['prob_1']:.3f})")
            print(f"  Text: {row['text']}")
            print(f"  Reason: {row['reason']}")
            print("-"*120)

        # Save corrections analysis
        corrections_df.to_csv('./submissions/corrections_analysis.csv', index=False)
        print(f"\nSaved corrections analysis to submissions/corrections_analysis.csv")

        # Apply corrections
        corrected_submission = apply_corrections(submission_df, corrections_df)

        print(f"\nCorrected submission stats:")
        print(corrected_submission['target'].value_counts())

        # Calculate changes
        original_1s = (submission_df['target'] == 1).sum()
        corrected_1s = (corrected_submission['target'] == 1).sum()
        print(f"\nChanges: {abs(corrected_1s - original_1s)} predictions modified")
        print(f"  0->1: {len(corrections_df[corrections_df['new_pred'] == 1])}")
        print(f"  1->0: {len(corrections_df[corrections_df['new_pred'] == 0])}")

        # Save corrected submission
        corrected_submission.to_csv('./submissions/submission_stage2_corrected.csv', index=False)
        print(f"\nSaved corrected submission to submissions/submission_stage2_corrected.csv")
    else:
        print("\nNo corrections found with current rules")


if __name__ == '__main__':
    main()
