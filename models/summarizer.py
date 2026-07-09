"""
Summarization Engine — The Core of the AI Text Summarizer
============================================================

📚 WHAT THIS MODULE DOES:
    This is the HEART of the entire application. It:
    1. Loads pretrained Transformer models (BART, T5, PEGASUS)
    2. Tokenizes input text into numeric IDs
    3. Runs the encoder-decoder model to generate summaries
    4. Decodes the output tokens back into human-readable text

📚 THE TRANSFORMER ARCHITECTURE (Deep Dive):
    ════════════════════════════════════════════

    The Transformer (2017, "Attention Is All You Need" paper) revolutionized NLP.
    Before Transformers, we used RNNs/LSTMs which process text sequentially
    (word by word). Transformers process ALL words simultaneously using "attention".

    ┌─────────────────────────────────────────────────────────────┐
    │                    ENCODER-DECODER FLOW                     │
    │                                                             │
    │   Input: "The quick brown fox jumps over the lazy dog"      │
    │                          │                                  │
    │                     [Tokenizer]                             │
    │                          │                                  │
    │         Token IDs: [464, 2068, 7586, 21831, ...]            │
    │                          │                                  │
    │              [Embedding Layer]                               │
    │    Each token ID → 768-dimensional vector                   │
    │                          │                                  │
    │          [Positional Encoding]                               │
    │    Add position information (sine/cosine waves)             │
    │    Position 1: [sin(1/10000^0), cos(1/10000^0), ...]        │
    │    WHY? Transformers have no built-in notion of order       │
    │                          │                                  │
    │              ┌───────────┴───────────┐                      │
    │              │                       │                      │
    │         [ENCODER]               [DECODER]                   │
    │     Reads full input          Generates output              │
    │     6-12 layers               token by token                │
    │              │                       │                      │
    │        Self-Attention          Cross-Attention               │
    │     "bank" attends to       Decoder attends to              │
    │     "river" → river bank    encoder output                  │
    │              │                       │                      │
    │              └───────────┬───────────┘                      │
    │                          │                                  │
    │                   [Linear + Softmax]                         │
    │         Probability distribution over vocabulary            │
    │                          │                                  │
    │              Output: "A fox jumped over a dog"              │
    └─────────────────────────────────────────────────────────────┘

    SELF-ATTENTION (The Key Innovation):
    ────────────────────────────────────
    For each token, attention computes:
        1. Query (Q): "What am I looking for?"
        2. Key (K):   "What do I contain?"
        3. Value (V): "What information do I provide?"

        Attention(Q, K, V) = softmax(QK^T / √d_k) × V

        The "bank" token creates a Query: "What context defines my meaning?"
        The "river" token's Key strongly matches → high attention weight
        So "bank" gets represented as "river bank" (not "financial bank")

    MULTI-HEAD ATTENTION:
    ────────────────────
    Multiple attention heads (typically 8-12) run in parallel.
    Each head learns to focus on different relationships:
        - Head 1: Syntactic (subject-verb agreement)
        - Head 2: Semantic (word meaning in context)
        - Head 3: Positional (nearby words)

📚 MODELS WE USE:

    BART (Bidirectional and Auto-Regressive Transformers):
    ─────────────────────────────────────────────────────
    - By Facebook AI (2019)
    - Trained by corrupting text (masking, deletion, permutation) then reconstructing
    - Encoder: Bidirectional (sees all input at once)
    - Decoder: Auto-regressive (generates left-to-right)
    - facebook/bart-large-cnn: Fine-tuned on CNN/DailyMail news articles

    T5 (Text-to-Text Transfer Transformer):
    ────────────────────────────────────────
    - By Google (2019)
    - Treats EVERY NLP task as text-to-text:
        Summarization: "summarize: <text>" → summary
        Translation:   "translate English to French: <text>" → French text
        Question:      "question: <q> context: <c>" → answer
    - t5-small: 60M parameters, fast, good for development

    PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarization):
    ──────────────────────────────────────────────────────────────────────────────────
    - By Google (2020)
    - Specifically designed for summarization
    - Pre-trained by masking important SENTENCES (not just words)
    - google/pegasus-xsum: Fine-tuned on XSum dataset (extreme single-sentence summaries)

📚 HUGGING FACE API:
    ─────────────────
    AutoTokenizer: Automatically selects the right tokenizer for any model
    AutoModelForSeq2SeqLM: Automatically selects the right model architecture
    pipeline(): Highest-level API — wraps tokenizer + model + postprocessing

    We use AutoTokenizer + AutoModelForSeq2SeqLM for fine-grained control,
    and also provide a pipeline() option for simplicity.

📚 GENERATION METHODS:
    ────────────────────
    Greedy Search: Always pick the most probable next token
        Pro: Fast, deterministic
        Con: Can get stuck in loops, misses better sequences

    Beam Search: Keep top-N candidate sequences at each step
        Pro: Explores multiple paths, finds better overall sequences
        Con: Slower, can produce generic/repetitive text

    Sampling (Top-K / Top-P):
        Top-K: Only sample from the K most probable tokens
        Top-P (Nucleus): Sample from tokens whose cumulative probability > p
        Pro: More diverse/creative output
        Con: Less reliable for factual summarization

    For summarization, we use BEAM SEARCH (num_beams=4) because
    we want reliable, faithful summaries — not creative writing.
"""

from dataclasses import dataclass
from typing import Optional
import logging
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    pipeline,
)

from config import GENERATION_DEFAULTS, ModelConfig
from models.model_registry import get_model_config, get_generation_params

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════

@dataclass
class SummaryResult:
    """
    Result of a summarization operation.

    Contains the summary text along with metadata that the UI displays:
    word counts, compression ratio, and timing information.
    """
    summary: str                   # The generated summary text
    original_word_count: int       # Word count of the input
    summary_word_count: int        # Word count of the summary
    compression_ratio: float       # original / summary (how much shorter)
    model_name: str                # Which model was used
    generation_time_seconds: float # How long inference took
    input_tokens: int              # Number of tokens in the input
    output_tokens: int             # Number of tokens in the output


# ══════════════════════════════════════════════
# SUMMARIZER CLASS
# ══════════════════════════════════════════════

class TextSummarizer:
    """
    Core summarization engine using Hugging Face Transformers.

    📚 DESIGN DECISIONS:

    1. Lazy Loading: Models are NOT loaded when TextSummarizer() is created.
       They're loaded on first use (load_model). This is important because:
       - Models are large (hundreds of MB to GB)
       - Loading takes seconds to minutes
       - The app starts faster if we load on demand

    2. Model Caching: Once loaded, models stay in memory (self._model, self._tokenizer).
       Reloading only happens when switching models.

    3. Device Selection: Automatically uses GPU (CUDA) if available, else CPU.
       GPU is 10-50x faster for inference.

    Usage:
        >>> summarizer = TextSummarizer()
        >>> summarizer.load_model("t5-small")
        >>> result = summarizer.summarize("Long text here...", summary_length="Medium")
        >>> print(result.summary)
        'A concise summary of the text...'
    """

    def __init__(self) -> None:
        """
        Initialize the summarizer.

        📚 torch.device EXPLAINED:
            PyTorch operations run on a "device":
            - "cpu": Uses your processor (always available, slower)
            - "cuda": Uses NVIDIA GPU (if available, much faster)
            - "mps": Uses Apple Silicon GPU (M1/M2/M3 Macs)

            torch.cuda.is_available() checks if a CUDA-capable GPU is present.
        """
        # Determine the best available device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            self._device = torch.device("cpu")
            logger.info("Using CPU (no GPU detected)")

        # Model and tokenizer (loaded lazily)
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._current_model_name: Optional[str] = None

    @property
    def device(self) -> torch.device:
        """The device models run on (CPU/GPU)."""
        return self._device

    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None and self._tokenizer is not None

    @property
    def current_model_name(self) -> Optional[str]:
        """Name of the currently loaded model."""
        return self._current_model_name

    def load_model(self, model_name: str) -> None:
        """
        Load a summarization model and its tokenizer.

        📚 AutoTokenizer.from_pretrained(model_name):
            - Downloads the tokenizer files from Hugging Face Hub (first time only)
            - Caches them locally (~/.cache/huggingface/)
            - Creates the correct tokenizer class automatically:
              BART → BartTokenizer, T5 → T5Tokenizer, etc.

        📚 AutoModelForSeq2SeqLM.from_pretrained(model_name):
            - Downloads model weights (first time only, can be 1-2 GB!)
            - Creates the correct model class
            - Loads weights into memory

        📚 model.to(device):
            Moves the model's parameters to the specified device (CPU/GPU).
            All tensors must be on the same device for operations to work.

        📚 model.eval():
            Sets the model to evaluation mode:
            - Disables dropout (random neuron deactivation during training)
            - Disables batch normalization updates
            - Makes inference deterministic and faster

        Args:
            model_name: Hugging Face model identifier.

        Raises:
            RuntimeError: If model loading fails.
        """
        # Skip if already loaded
        if self._current_model_name == model_name and self.is_model_loaded:
            logger.info("Model '%s' is already loaded", model_name)
            return

        logger.info("Loading model: %s", model_name)
        start_time = time.time()

        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded: %s", type(self._tokenizer).__name__)

            # Load model
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info("Model loaded: %s", type(self._model).__name__)

            # Move to device and set to evaluation mode
            self._model.to(self._device)
            self._model.eval()

            self._current_model_name = model_name

            elapsed = time.time() - start_time
            logger.info(
                "Model '%s' loaded successfully in %.1f seconds on %s",
                model_name, elapsed, self._device,
            )

        except Exception as e:
            self._model = None
            self._tokenizer = None
            self._current_model_name = None
            logger.error("Failed to load model '%s': %s", model_name, e)
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def summarize(
        self,
        text: str,
        summary_length: str = "Medium",
        **kwargs,
    ) -> SummaryResult:
        """
        Generate a summary of the input text.

        📚 THE SUMMARIZATION PIPELINE (Step by Step):

        1. TOKENIZATION:
           "The cat sat on the mat" → [464, 3797, 3332, 373, 262, 2603]
           The tokenizer converts text to token IDs that the model understands.

           For T5, we prepend "summarize: " to the input:
           "summarize: The cat sat on the mat" → [21603, 10, 464, 3797, ...]

        2. ENCODING:
           Token IDs → Tensor → Model's encoder → Contextual representations
           Each token gets a 768-dim vector that captures its meaning IN CONTEXT.

        3. GENERATION (Beam Search):
           Starting from a <START> token, the decoder generates one token at a time:
           Step 1: <START> → "A" (most probable first token)
           Step 2: <START> A → "cat" (most probable given "A")
           Step 3: <START> A cat → "rested" (paraphrased!)
           ... until <END> token or max_length reached

           With num_beams=4, we explore 4 paths simultaneously and pick the best.

        4. DECODING:
           Output token IDs → Human-readable text
           [32, 3797, 22823] → "A cat rested"

        Args:
            text: Input text to summarize.
            summary_length: "Short", "Medium", or "Long".
            **kwargs: Additional generation parameters (override defaults).

        Returns:
            SummaryResult with the summary and metadata.

        Raises:
            RuntimeError: If no model is loaded.
            ValueError: If input text is too short.
        """
        if not self.is_model_loaded:
            raise RuntimeError(
                "No model loaded. Call load_model() first."
            )

        if len(text.split()) < 10:
            raise ValueError(
                "Input text is too short (less than 10 words). "
                "Please provide more text to summarize."
            )

        logger.info(
            "Summarizing %d words with %s (%s length)",
            len(text.split()), self._current_model_name, summary_length,
        )

        start_time = time.time()

        # Step 1: Prepare input text
        input_text = self._prepare_input(text)

        # Step 2: Tokenize
        model_config = get_model_config(self._current_model_name)
        inputs = self._tokenizer(
            input_text,
            max_length=model_config.max_input_length,
            truncation=True,    # Cut text if too long
            padding="longest",  # Pad shorter sequences in a batch
            return_tensors="pt",  # Return PyTorch tensors (not lists)
        )
        # 📚 return_tensors="pt":
        #    "pt" = PyTorch tensors. Other options: "tf" (TensorFlow), "np" (NumPy)
        #    Models expect tensor inputs, not Python lists.

        # Move input tensors to the same device as the model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_token_count = inputs["input_ids"].shape[1]

        # Step 3: Get generation parameters
        gen_params = get_generation_params(self._current_model_name, summary_length)
        gen_params.update(kwargs)  # Allow overrides

        # Step 4: Generate summary
        # 📚 torch.no_grad():
        #    Disables gradient computation. During inference, we don't need
        #    gradients (those are only for training). This saves memory and
        #    makes inference ~20% faster.
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_params,
            )

        # Step 5: Decode output tokens back to text
        summary = self._tokenizer.decode(
            output_ids[0],          # First (and only) sequence in the batch
            skip_special_tokens=True,  # Remove <pad>, <eos>, etc.
            clean_up_tokenization_spaces=True,
        )

        # Compute metrics
        elapsed = time.time() - start_time
        original_word_count = len(text.split())
        summary_word_count = len(summary.split())
        compression_ratio = (
            original_word_count / summary_word_count
            if summary_word_count > 0 else 0.0
        )

        result = SummaryResult(
            summary=summary.strip(),
            original_word_count=original_word_count,
            summary_word_count=summary_word_count,
            compression_ratio=round(compression_ratio, 2),
            model_name=self._current_model_name,
            generation_time_seconds=round(elapsed, 2),
            input_tokens=input_token_count,
            output_tokens=output_ids.shape[1],
        )

        logger.info(
            "Summary generated: %d → %d words (%.1fx compression) in %.1fs",
            result.original_word_count,
            result.summary_word_count,
            result.compression_ratio,
            result.generation_time_seconds,
        )

        return result

    def _prepare_input(self, text: str) -> str:
        """
        Prepare input text for the specific model.

        📚 T5 REQUIRES A TASK PREFIX:
            T5 was trained on multiple tasks with prefixes:
            - "summarize: <text>"
            - "translate English to German: <text>"
            - "question: <q> context: <c>"

            Without the prefix, T5 doesn't know what task to perform!
            BART and PEGASUS don't need prefixes — they're fine-tuned for one task.
        """
        if self._current_model_name and "t5" in self._current_model_name.lower():
            return f"summarize: {text}"
        return text

    def summarize_with_pipeline(
        self,
        text: str,
        summary_length: str = "Medium",
    ) -> SummaryResult:
        """
        Alternative summarization using Hugging Face pipeline() API.

        📚 pipeline() vs Manual Approach:
            pipeline() is a high-level wrapper that handles:
            - Tokenization
            - Model inference
            - Decoding
            All in one line! Great for prototyping.

            But the manual approach (our main summarize method) gives us:
            - More control over parameters
            - Better error handling
            - Ability to log intermediate steps
            - Custom preprocessing

            In production, use the manual approach. For quick experiments, use pipeline.

        Args:
            text: Input text to summarize.
            summary_length: "Short", "Medium", or "Long".

        Returns:
            SummaryResult with the summary and metadata.
        """
        if not self.is_model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start_time = time.time()
        gen_params = get_generation_params(self._current_model_name, summary_length)

        input_text = self._prepare_input(text)

        # Create pipeline
        summarizer_pipeline = pipeline(
            "summarization",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
        )

        # Generate summary
        result = summarizer_pipeline(
            input_text,
            max_length=gen_params["max_length"],
            min_length=gen_params["min_length"],
            num_beams=gen_params["num_beams"],
            length_penalty=gen_params.get("length_penalty", 2.0),
            early_stopping=gen_params.get("early_stopping", True),
            no_repeat_ngram_size=gen_params.get("no_repeat_ngram_size", 3),
        )

        summary = result[0]["summary_text"]
        elapsed = time.time() - start_time

        original_word_count = len(text.split())
        summary_word_count = len(summary.split())
        compression_ratio = (
            original_word_count / summary_word_count
            if summary_word_count > 0 else 0.0
        )

        return SummaryResult(
            summary=summary.strip(),
            original_word_count=original_word_count,
            summary_word_count=summary_word_count,
            compression_ratio=round(compression_ratio, 2),
            model_name=self._current_model_name,
            generation_time_seconds=round(elapsed, 2),
            input_tokens=0,   # Pipeline doesn't expose this
            output_tokens=0,
        )

    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.

        Useful for display in the UI and debugging.
        """
        if not self.is_model_loaded:
            return {"status": "No model loaded"}

        # Count parameters
        total_params = sum(p.numel() for p in self._model.parameters())
        size_mb = sum(
            p.numel() * p.element_size() for p in self._model.parameters()
        ) / (1024 * 1024)

        return {
            "model_name": self._current_model_name,
            "model_class": type(self._model).__name__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "total_parameters": f"{total_params:,}",
            "model_size_mb": f"{size_mb:.1f} MB",
            "device": str(self._device),
            "vocabulary_size": self._tokenizer.vocab_size,
        }

    def unload_model(self) -> None:
        """
        Unload the current model to free memory.

        📚 MEMORY MANAGEMENT:
            Large models consume significant RAM/VRAM.
            Explicitly deleting and garbage collecting frees this memory.
            torch.cuda.empty_cache() releases GPU memory back to the system.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._current_model_name = None

        # Force garbage collection and clear GPU cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory freed")
