# LLM API-based Few-Shot Learning for Disaster Tweets

This implementation uses external Large Language Model (LLM) APIs with carefully designed few-shot learning prompts to classify disaster tweets.

## Approach

Unlike fine-tuning approaches (12.7 BERT and 12.15 Qwen LoRA), this method leverages the powerful reasoning capabilities of large pre-trained models through:

1. **Few-Shot Learning**: Provides 8 carefully selected examples (4 disaster, 4 non-disaster) in the prompt
2. **Prompt Engineering**: Detailed system prompt explaining the classification criteria
3. **API-based Inference**: No model training required, uses external LLM API

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Inference

```bash
# Basic usage (8-shot learning)
python run_api_inference.py

# With validation evaluation
python run_api_inference.py --evaluate --val_size 100

# Custom configuration
python run_api_inference.py \
    --n_examples 16 \
    --temperature 0.2 \
    --batch_delay 0.3 \
    --model deepseek-v3
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_examples` | 8 | Number of few-shot examples |
| `--batch_delay` | 0.5 | Delay between API calls (seconds) |
| `--temperature` | 0.1 | LLM temperature (lower = more deterministic) |
| `--evaluate` | False | Evaluate on validation set first |
| `--val_size` | 100 | Validation set size |
| `--model` | deepseek-v3 | LLM model to use |

## Key Features

### 1. Intelligent Prompt Design

The system prompt clearly defines what constitutes a disaster tweet:
- Natural disasters (earthquakes, floods, etc.)
- Accidents (crashes, explosions, etc.)
- Emergency situations
- Deaths/casualties from disasters
- Urgent warnings

And what doesn't:
- Metaphorical usage
- Fiction/movies
- Hyperbole
- Non-urgent historical references

### 2. Balanced Few-Shot Examples

Automatically selects balanced examples (equal disaster/non-disaster) from training data to provide representative context.

### 3. Robust Error Handling

- Automatic retry on API failures
- Rate limiting protection
- Response parsing with fallbacks

### 4. Evaluation Support

Can evaluate on validation set before running full test inference to tune parameters.

## Advantages

- **No Training Required**: Instant deployment, no GPU needed
- **Flexibility**: Easy to adjust prompts and examples
- **Interpretability**: Clear reasoning through prompt design
- **Cost-Effective**: Pay-per-use pricing model

## Limitations

- **API Dependency**: Requires internet and API access
- **Latency**: Slower than local inference
- **Cost**: Per-token pricing on large test sets
- **Rate Limits**: Need to manage API rate limits

## Results

Results will be saved in `../submissions/` with format:
```
llm_fewshot_{model}_{n_examples}shot_{timestamp}.csv
llm_fewshot_{model}_{n_examples}shot_{timestamp}_config.json
```

The config file contains the exact parameters and few-shot examples used for reproducibility.
