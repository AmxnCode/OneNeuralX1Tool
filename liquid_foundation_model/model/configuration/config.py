from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class X1ToolConfig:
    """
    Configuration for One Neural X1 Tool - Encoder-Decoder model for function calling.
    
    Architecture:
    - Encoder: 12 layers, GQA (8H/4KV), RoPE, gated residuals
    - Decoder: 8 layers, self-attn + cross-attn, gated residuals
    - d_model: 512
    - Vocab: 8,192 (SentencePiece BPE)
    - No FFN (pure attention)
    - Total: ~27M parameters
    
    Based on Needle architecture by Cactus-Compute.
    """
    
    # Model type
    model_type: str = "one_neural_x1_tool"
    
    # Vocabulary
    vocab_size: int = 8192
    
    # Hidden size
    hidden_size: int = 512
    
    # Encoder configuration
    num_encoder_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    
    # Decoder configuration
    num_decoder_layers: int = 8
    
    # Attention
    head_dim: Optional[int] = None  # Computed as hidden_size // num_attention_heads
    
    # Sequence length
    max_seq_len: int = 8192
    
    # RoPE
    rope_theta: float = 10000.0
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Embedding tying
    tie_embeddings: bool = True
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Function calling tokens
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_result_start: str = "<tool_result>"
    tool_result_end: str = "</tool_result>"
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
    
    @classmethod
    def small(cls) -> "X1ToolConfig":
        """Small model (~27M params) - matches Needle."""
        return cls(
            vocab_size=8192,
            hidden_size=512,
            num_encoder_layers=12,
            num_decoder_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_seq_len=8192,
        )
    
    @classmethod
    def tiny(cls) -> "X1ToolConfig":
        """Tiny model (~10M params) - for testing."""
        return cls(
            vocab_size=8192,
            hidden_size=256,
            num_encoder_layers=6,
            num_decoder_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=4096,
        )
    
    @classmethod
    def medium(cls) -> "X1ToolConfig":
        """Medium model (~100M params) - if scaling up."""
        return cls(
            vocab_size=16384,
            hidden_size=1024,
            num_encoder_layers=16,
            num_decoder_layers=12,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_seq_len=16384,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "X1ToolConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class LFMConfig:
    """Configuration class for Liquid Foundation Model."""
    
    # Model size and architecture
    model_size: Literal["350M", "700M", "1.2B"] = "350M"
    num_layers: int = 16  # num_hidden_layers in config.json
    hidden_size: int = 1024  # hidden_size in config.json
    intermediate_size: int = 6656  # block_ff_dim in config.json
    num_attention_heads: int = 16  # num_attention_heads in config.json
    num_key_value_heads: int = 8  # num_key_value_heads in config.json
    max_position_embeddings: int = 128000  # max_position_embeddings in config.json
    vocab_size: int = 65536  # vocab_size in config.json
    
    # Convolution parameters
    conv_kernel_size: int = 4
    conv_bias: bool = False  # conv_bias in config.json
    conv_dim: int = 1024  # conv_dim in config.json
    conv_dim_out: int = 1024  # conv_dim_out in config.json
    conv_L_cache: int = 3  # conv_L_cache in config.json
    
    # Block structure
    # The official model uses full_attn_idxs to specify which blocks are attention blocks
    full_attn_idxs: List[int] = None  # Will be set in __post_init__
    num_conv_blocks: int = 10  # Derived from full_attn_idxs
    num_attention_blocks: int = 6  # Derived from full_attn_idxs
    
    # Regularization
    dropout_rate: float = 0.0  # Not explicitly specified in config.json
    attention_dropout_rate: float = 0.0  # Not explicitly specified in config.json
    layer_norm_epsilon: float = 1e-5  # norm_eps in config.json
    
    # Activation function
    hidden_act: str = "swiglu"  # block_use_swiglu is True in config.json
    
    # Initialization
    initializer_range: float = 0.02  # initializer_range in config.json
    block_mlp_init_scale: float = 1.0  # block_mlp_init_scale in config.json
    block_out_init_scale: float = 1.0  # block_out_init_scale in config.json
    block_use_xavier_init: bool = True  # block_use_xavier_init in config.json
    conv_use_xavier_init: bool = True  # conv_use_xavier_init in config.json
    
    # HRM parameters
    enable_hrm: bool = True
    hrm_planning_size: Optional[int] = None  # Will default to hidden_size // 2
    hrm_detail_size: Optional[int] = None    # Will default to hidden_size
    hrm_reasoning_steps: int = 3
    num_task_types: int = 3  # general, complex, reasoning
    
    # Other parameters
    tie_word_embeddings: bool = False  # Not specified in config.json
    pad_token_id: int = 0  # pad_token_id in config.json
    bos_token_id: int = 1  # bos_token_id in config.json
    eos_token_id: int = 7  # eos_token_id in config.json
    use_cache: bool = True  # use_cache in config.json
    use_pos_enc: bool = True  # use_pos_enc in config.json
    rope_theta: float = 1000000.0  # rope_theta in config.json
    torch_dtype: str = "bfloat16"  # torch_dtype in config.json
    
    def __post_init__(self):
        # Set default full_attn_idxs if not provided
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [2, 5, 8, 10, 12, 14]  # From config.json
    
    @classmethod
    def from_pretrained(cls, model_size: str) -> "LFMConfig":
        """Create a configuration for a specific model size."""
        if model_size == "350M" or model_size == "small":
            return cls(
                model_size="350M",
                num_layers=16,
                hidden_size=1024,
                intermediate_size=6656,
                num_attention_heads=16,
                num_key_value_heads=8,
                full_attn_idxs=[2, 5, 8, 10, 12, 14],
                vocab_size=65536,
            )
        elif model_size == "700M" or model_size == "medium":
            # These values are estimated based on scaling up from 350M
            return cls(
                model_size="700M",
                num_layers=16,
                hidden_size=1536,  # Scaled up from 350M
                intermediate_size=9984,  # Scaled up from 350M
                num_attention_heads=24,  # Scaled up from 350M
                num_key_value_heads=8,
                full_attn_idxs=[2, 5, 8, 10, 12, 14],
                vocab_size=65536,
                conv_dim=1536,
                conv_dim_out=1536,
            )
        elif model_size == "1.2B" or model_size == "large":
            # These values are estimated based on scaling up from 350M
            return cls(
                model_size="1.2B",
                num_layers=16,
                hidden_size=2048,  # Scaled up from 350M
                intermediate_size=13312,  # Scaled up from 350M
                num_attention_heads=32,  # Scaled up from 350M
                num_key_value_heads=8,
                full_attn_idxs=[2, 5, 8, 10, 12, 14],
                vocab_size=65536,
                conv_dim=2048,
                conv_dim_out=2048,
            )
        else:
            raise ValueError(f"Unknown model size: {model_size}")
    
    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "LFMConfig":
        """Create a configuration from a dictionary."""
        return cls(**config_dict)


@dataclass
class X1ToolTrainingConfig:
    """
    Training configuration for One Neural X1 Tool.
    
    Optimized for function calling / tool use training.
    """
    
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 200000
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Optimization
    gradient_checkpointing: bool = True
    
    # Knowledge distillation
    distillation_alpha: float = 0.5
    
    # DPO
    dpo_beta: float = 0.1
    
    # Function calling specific
    function_call_loss_weight: float = 1.0
    structured_output_loss_weight: float = 0.5
    
    # Data
    max_seq_len: int = 8192
    
    # Distillation
    teacher_model: str = "claude-3-5-haiku"  # or "gpt-4o-mini", "gemini-flash"
    
    @classmethod
    def pretrain(cls) -> "X1ToolTrainingConfig":
        """Pre-training configuration (200B tokens)."""
        return cls(
            batch_size=256,
            learning_rate=3e-4,
            warmup_steps=10000,
            max_steps=1000000,  # ~200B tokens with batch_size=256 * seq_len=8192
            gradient_accumulation_steps=4,
        )
    
    @classmethod
    def distill(cls) -> "X1ToolTrainingConfig":
        """Distillation configuration."""
        return cls(
            batch_size=64,
            learning_rate=1e-4,
            warmup_steps=500,
            max_steps=50000,
            distillation_alpha=0.7,
        )
    
    @classmethod
    def sft(cls) -> "X1ToolTrainingConfig":
        """SFT configuration for function calling."""
        return cls(
            batch_size=32,
            learning_rate=2e-5,
            warmup_steps=100,
            max_steps=10000,
            function_call_loss_weight=2.0,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingConfig:
    """Configuration class for training the Liquid Foundation Model."""
    
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    
    # Mixed precision training
    fp16: bool = False
    bf16: bool = True  # Updated to bfloat16 as used in official models
    
    # Optimization
    gradient_checkpointing: bool = False
    
    # Knowledge distillation
    distillation_alpha: float = 0.5  # Weight for distillation loss
    
    # DPO
    dpo_beta: float = 0.1  # DPO hyperparameter


@dataclass
class InferenceConfig:
    """Configuration class for inference with the Liquid Foundation Model."""
    
    # Device and quantization
    device: str = "cpu"  # "cpu", "cuda", "mps"
    quantization: Optional[str] = None  # "8bit", "4bit", None
    
    # Generation parameters
    batch_size: int = 1
    max_length: int = 32768  # Based on context length from model card
    num_beams: int = 1
    temperature: float = 0.3  # Based on recommended settings from model card
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.15  # Added based on recommended settings from model card
    repetition_penalty: float = 1.05  # Added based on recommended settings from model card