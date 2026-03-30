"""Configuration for baseline experiments on HotpotQA."""

# ============================================================================
# Model Configuration
# ============================================================================

MODELS = [
    "gpt-4o-mini",
    "gpt-5.4-mini",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
]

# ============================================================================
# LLM Provider & Pricing Configuration
# ============================================================================
# Maps each model to its provider and pricing info (per 1M tokens)
# Last validated: 2026-03-29

LLM_CONFIG = {
    "gpt-4o-mini": {
        "provider": "openai",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "input_price_per_1m": 0.20,
        "output_price_per_1m": 0.80,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "input_price_per_1m": 0.80,
        "output_price_per_1m": 4.00,
        "cache_read_price_per_1m": 0.24,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "cache_read_price_per_1m": 0.90,
    },
}

# Provider grouping for routing
PROVIDER_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-5.4-mini"],
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-4-6"],
}

# ============================================================================
# Baseline Methods
# ============================================================================

METHODS = [
    "always_think",      # No retrieval, pure reasoning
    "retrieval_k3",      # Retrieve top-3 by embedding similarity
    "retrieval_k5",      # Retrieve top-5 by embedding similarity
    "retrieval_k10",     # Retrieve top-10 by embedding similarity
]

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_CONFIG = {
    "dataset_name": "hotpot_qa",
    "subset": "distractor",  # Multi-hop reasoning with distractor paragraphs
    "split": "validation",   # Use validation split
}

# ============================================================================
# LLM Model to Embedding Config Mapping
# ============================================================================
# Maps each LLM model to its embedding provider, model, and max token capacity
# Includes the embedding model's native maximum for full context preservation
# Last validated: 2026-03-29

LLM_TO_EMBEDDING = {
    "gpt-4o-mini": {
        "provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "max_tokens": 8191,  # OpenAI text-embedding-3-small max
    },
    "gpt-5.4-mini": {
        "provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "max_tokens": 8191,  # OpenAI text-embedding-3-small max
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "embedding_model": "voyage-3",
        "max_tokens": 32000,  # Voyage AI voyage-3 max per text
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "embedding_model": "voyage-3",
        "max_tokens": 32000,  # Voyage AI voyage-3 max per text
    },
}

# ============================================================================
# Evaluation Sample Sizes
# ============================================================================

# Exploratory runs — for validating code is working
EXPLORATORY_N_ROWS = [2, 10]

# Production runs — for final evaluation
PRODUCTION_N_ROWS = [100, 200, 300, 400]

# ============================================================================
# Output Directory
# ============================================================================
# Results are written to experiments/results/baseline/
# Each model gets two files:
#   - baseline_{model}.csv  (per-record metrics)
#   - baseline_{model}.md   (summary report with global metrics)
