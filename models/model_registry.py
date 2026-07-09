"""
Model Registry — Centralized Model Configuration & Management
================================================================

📚 WHAT THIS MODULE DOES:
    Manages the registry of available summarization models.
    Provides functions to:
    - Look up model configurations by name
    - List available models for the UI
    - Validate model names
    - Map summary length presets to model-specific parameters

📚 WHY A SEPARATE REGISTRY?
    The summarizer module (summarizer.py) handles model LOADING and INFERENCE.
    The registry handles model METADATA and CONFIGURATION.

    Separation of concerns:
    - Registry knows: "BART expects max_length=142 for news articles"
    - Summarizer knows: "Load BART, tokenize input, generate output"

    This makes it easy to add new models without touching the summarizer logic.
"""

import logging
from typing import Optional

from config import AVAILABLE_MODELS, DEFAULT_MODEL_NAME, SUMMARY_LENGTHS, ModelConfig

logger = logging.getLogger(__name__)


def get_model_config(model_name: str) -> ModelConfig:
    """
    Retrieve the configuration for a specific model.

    Args:
        model_name: Hugging Face model identifier (e.g., "facebook/bart-large-cnn").

    Returns:
        ModelConfig dataclass with the model's settings.

    Raises:
        ValueError: If the model name is not in the registry.

    Example:
        >>> config = get_model_config("t5-small")
        >>> config.display_name
        'T5 Small (Fast, Lightweight)'
        >>> config.max_input_length
        512
    """
    for model in AVAILABLE_MODELS:
        if model.name == model_name:
            return model

    available = [m.name for m in AVAILABLE_MODELS]
    raise ValueError(
        f"Unknown model: '{model_name}'. Available models: {available}"
    )


def get_default_model() -> ModelConfig:
    """Return the default model configuration."""
    return get_model_config(DEFAULT_MODEL_NAME)


def list_models() -> list[ModelConfig]:
    """Return all available model configurations."""
    return list(AVAILABLE_MODELS)


def get_model_names() -> list[str]:
    """Return just the model name strings (for dropdowns)."""
    return [m.name for m in AVAILABLE_MODELS]


def get_display_names() -> dict[str, str]:
    """
    Return a mapping of display_name → model_name.
    Used for Streamlit selectbox where user sees display names.
    """
    return {m.display_name: m.name for m in AVAILABLE_MODELS}


def get_generation_params(
    model_name: str,
    summary_length: str = "Medium",
) -> dict:
    """
    Get generation parameters tailored to a specific model and summary length.

    📚 WHY MODEL-SPECIFIC PARAMETERS?
        Different models were trained with different settings:
        - BART-CNN: Trained on news summaries averaging 56 tokens → min_length=56
        - PEGASUS-XSum: Trained on 1-sentence summaries → max_length=64
        - T5: General-purpose → flexible parameters

        Using the wrong parameters produces poor summaries.

    Args:
        model_name: Hugging Face model identifier.
        summary_length: "Short", "Medium", or "Long".

    Returns:
        Dictionary of generation parameters ready for model.generate().
    """
    model_config = get_model_config(model_name)
    length_preset = SUMMARY_LENGTHS.get(summary_length, SUMMARY_LENGTHS["Medium"])

    params = {
        "max_length": length_preset.max_length,
        "min_length": length_preset.min_length,
        "num_beams": model_config.default_num_beams,
        "length_penalty": 2.0,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,  # Prevents repeating 3-word phrases
    }

    logger.debug(
        "Generation params for %s (%s): %s",
        model_name, summary_length, params,
    )

    return params
