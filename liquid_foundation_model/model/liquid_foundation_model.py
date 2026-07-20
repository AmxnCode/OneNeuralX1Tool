import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union

from liquid_foundation_model.model.configuration.config import LFMConfig
from liquid_foundation_model.model.blocks.conv_block import ConvBlock
from liquid_foundation_model.model.blocks.attention_block import AttentionBlock
from liquid_foundation_model.model.layers.rmsnorm import RMSNorm
from liquid_foundation_model.model.layers.hrm_module import HierarchicalReasoningModule, TaskRouter


class LiquidFoundationModel(nn.Module):
    """
    Liquid Foundation Model implementation.
    
    This model uses a hybrid architecture with convolution and attention blocks,
    inspired by the LFM2 architecture. The model consists of 16 blocks total,
    with attention blocks placed at specific indices defined in config.full_attn_idxs
    and convolution blocks at all other positions.
    """
    
    def __init__(self, config: LFMConfig):
        """
        Initialize the Liquid Foundation Model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Blocks
        self.blocks = nn.ModuleList()
        
        # Create blocks based on configuration
        # Use full_attn_idxs to determine which blocks are attention blocks
        for i in range(config.num_layers):
            if i in config.full_attn_idxs:
                # Create attention block
                self.blocks.append(
                    AttentionBlock(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.intermediate_size,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        dropout_rate=config.dropout_rate,
                        attention_dropout_rate=config.attention_dropout_rate,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                    )
                )
            else:
                # Create convolution block
                self.blocks.append(
                    ConvBlock(
                        hidden_size=config.hidden_size,
                        intermediate_size=config.intermediate_size,
                        kernel_size=config.conv_kernel_size,
                        dropout_rate=config.dropout_rate,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                    )
                )
        
        # HRM integration
        self.enable_hrm = getattr(config, 'enable_hrm', False)
        if self.enable_hrm:
            self.hrm_module = HierarchicalReasoningModule(
                hidden_size=config.hidden_size,
                planning_size=getattr(config, 'hrm_planning_size', config.hidden_size // 2),
                detail_size=getattr(config, 'hrm_detail_size', config.hidden_size),
                num_reasoning_steps=getattr(config, 'hrm_reasoning_steps', 3)
            )
            
            self.task_router = TaskRouter(
                hidden_size=config.hidden_size,
                num_task_types=getattr(config, 'num_task_types', 3)
            )
            
            # HRM integration layers
            self.hrm_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.hrm_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Final layer norm
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store model size information
        self._calculate_params()
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with small random values
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        use_hrm: Optional[bool] = None,
        task_type: str = "auto",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            If output_hidden_states is False:
                hidden_states: Final hidden states of shape (batch_size, seq_len, hidden_size)
            If output_hidden_states is True:
                hidden_states: Final hidden states of shape (batch_size, seq_len, hidden_size)
                all_hidden_states: List of hidden states from all layers
        """
        batch_size, seq_length = input_ids.shape
        
        # Create attention mask for attention blocks
        if attention_mask is not None:
            # Convert attention mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            # Convert from 0/1 to -inf/0
            attention_mask = attention_mask.to(dtype=self.embed_tokens.weight.dtype)  # fp32 or fp16
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Store all hidden states if required
        all_hidden_states = [] if output_hidden_states else None
        
        # Apply blocks
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            if i in self.config.full_attn_idxs:
                hidden_states = block(hidden_states, attention_mask)
            else:
                hidden_states = block(hidden_states)
        
        # HRM integration
        if self.enable_hrm:
            # Determine whether to use HRM
            should_use_hrm = self._should_use_hrm(hidden_states, task_type, use_hrm)
            
            if should_use_hrm:
                # Apply hierarchical reasoning
                hrm_output = self.hrm_module(hidden_states, use_reasoning=True)
                
                # Gate HRM output with original
                combined = torch.cat([hidden_states, hrm_output], dim=-1)
                gate = torch.sigmoid(self.hrm_gate(combined))
                hidden_states = gate * hrm_output + (1 - gate) * hidden_states
                hidden_states = self.hrm_norm(hidden_states)
        
        # Apply final normalization
        hidden_states = self.norm_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        else:
            return hidden_states
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings module."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module):
        """Set the input embeddings module."""
        self.embed_tokens = value
        
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings."""
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        
        # Update vocab_size in config
        self.config.vocab_size = new_num_tokens
        
        return self.get_input_embeddings()
    
    def _get_resized_embeddings(self, old_embeddings: nn.Embedding, new_num_tokens: int):
        """Get new embeddings when resizing token embeddings."""
        if new_num_tokens == old_embeddings.num_embeddings:
            return old_embeddings
        
        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy token embeddings from the previous weights
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        return new_embeddings
    
    def _calculate_params(self):
        """Calculate and store the number of parameters in the model."""
        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.num_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Convert to millions for easier reading
        self.num_parameters_millions = self.num_parameters / 1_000_000
        self.num_trainable_parameters_millions = self.num_trainable_parameters / 1_000_000
    
    def _should_use_hrm(self, hidden_states: torch.Tensor, task_type: str, use_hrm: Optional[bool]) -> bool:
        """Determine whether to use HRM based on task type and complexity."""
        if use_hrm is not None:
            return use_hrm
        
        if task_type == "reasoning":
            return True
        elif task_type == "general":
            return False
        elif task_type == "auto":
            # Use task router to automatically determine
            task_logits, _ = self.task_router(hidden_states)
            # Use HRM if classified as complex reasoning task (class 2)
            predicted_task = torch.argmax(task_logits, dim=-1)
            return (predicted_task == 2).any().item()
        
        return False
    
    def get_model_size_info(self):
        """Get information about the model size."""
        return {
            "model_size": self.config.model_size,
            "num_parameters": self.num_parameters,
            "num_parameters_millions": f"{self.num_parameters_millions:.2f}M",
            "num_trainable_parameters": self.num_trainable_parameters,
            "num_trainable_parameters_millions": f"{self.num_trainable_parameters_millions:.2f}M",
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "num_attention_blocks": len(self.config.full_attn_idxs),
            "num_conv_blocks": self.config.num_layers - len(self.config.full_attn_idxs),
            "hrm_enabled": self.enable_hrm,
        }


class LiquidFoundationModelForCausalLM(nn.Module):
    """
    Liquid Foundation Model for Causal Language Modeling.
    
    This adds a language modeling head on top of the base model.
    """
    
    def __init__(self, config: LFMConfig):
        """
        Initialize the model for causal language modeling.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Base model
        self.model = LiquidFoundationModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with small random values
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            
    def get_input_embeddings(self):
        """Get the input embeddings module."""
        return self.model.get_input_embeddings()
        
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings for both the base model and the language modeling head."""
        # Resize base model embeddings
        self.model.resize_token_embeddings(new_num_tokens)
        
        # Resize language modeling head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.config.hidden_size, new_num_tokens, bias=False)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        
        # Copy weights for existing tokens
        num_tokens_to_copy = min(old_lm_head.out_features, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        
        self.lm_head = new_lm_head
        
        # Update vocab_size in config
        self.config.vocab_size = new_num_tokens
        
        # Re-tie weights if needed
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
        return self.model.get_input_embeddings()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        use_hrm: Optional[bool] = None,
        task_type: str = "auto",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len)
            labels: Labels for language modeling of shape (batch_size, seq_len)
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            If labels is None:
                logits: Logits of shape (batch_size, seq_len, vocab_size)
            If labels is not None:
                loss: Language modeling loss
                logits: Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Get hidden states from base model with HRM support
        if output_hidden_states:
            hidden_states, all_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
                use_hrm=use_hrm,
                task_type=task_type,
            )
        else:
            hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
                use_hrm=use_hrm,
                task_type=task_type,
            )
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        
        if output_hidden_states:
            return (loss, logits, all_hidden_states) if loss is not None else (logits, all_hidden_states)
        else:
            return (loss, logits) if loss is not None else logits
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: int = None,
        eos_token_id: int = None,
        bos_token_id: int = None,
        use_cache: bool = True,
        stopping_criteria: Optional[List[callable]] = None,
        return_dict_in_generate: bool = False,
    ) -> Union[torch.LongTensor, dict]:
        """
        Generate text using the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequences
            min_length: Minimum length of generated sequences
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            bos_token_id: ID of the beginning-of-sequence token
            use_cache: Whether to use past key values for faster generation
            stopping_criteria: List of callables that determine when to stop generation
            return_dict_in_generate: Whether to return a dictionary with additional information
            
        Returns:
            If return_dict_in_generate is False:
                generated_ids: Generated token IDs of shape (batch_size * num_return_sequences, max_length)
            If return_dict_in_generate is True:
                A dictionary containing:
                - sequences: Generated token IDs
                - scores: Token scores at each step (if do_sample is True)
                - attentions: Attention weights at each step (if output_attentions is True)
                - hidden_states: Hidden states at each step (if output_hidden_states is True)
        """
        # Set default values
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize stopping criteria if not provided
        if stopping_criteria is None:
            stopping_criteria = []
        
        # Add default stopping criteria
        class MaxLengthCriteria:
            def __init__(self, max_length):
                self.max_length = max_length
            
            def __call__(self, input_ids, scores, **kwargs):
                return input_ids.shape[-1] >= self.max_length
        
        class EosTokenCriteria:
            def __init__(self, eos_token_id):
                self.eos_token_id = eos_token_id
            
            def __call__(self, input_ids, scores, **kwargs):
                # Check if all sequences have generated EOS token
                if self.eos_token_id is None:
                    return False
                return (input_ids[:, -1] == self.eos_token_id).all()
        
        stopping_criteria.append(MaxLengthCriteria(max_length))
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id))
        
        # Expand input for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        # Initialize generated sequences with input_ids
        generated_ids = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Initialize cache for key/value states if using cache
        past_key_values = None
        
        # Initialize storage for scores if returning dict
        scores = [] if return_dict_in_generate and do_sample else None
        
        # Generate tokens
        while not any(criterion(generated_ids, None) for criterion in stopping_criteria):
            # Forward pass
            with torch.no_grad():
                if use_cache and past_key_values is not None:
                    # Only process the last token with cached key/values
                    current_input_ids = generated_ids[:, -1].unsqueeze(-1)
                    current_attention_mask = attention_mask
                else:
                    # Process the entire sequence
                    current_input_ids = generated_ids
                    current_attention_mask = attention_mask
                
                outputs = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                )
                
                # Get logits for the next token
                if isinstance(outputs, tuple):
                    next_token_logits = outputs[0][:, -1, :]
                else:
                    next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size * num_return_sequences):
                        for token_id in set(generated_ids[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Prevent EOS token before min_length
                if min_length > 0 and generated_ids.shape[1] < min_length:
                    next_token_logits[:, eos_token_id] = float('-inf')
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy decode
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    
                    # Store scores if returning dict
                    if scores is not None:
                        scores.append(next_token_logits)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append next tokens to generated_ids
                generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens.unsqueeze(-1))], dim=-1)
        
        # Return results
        if return_dict_in_generate:
            result = {
                "sequences": generated_ids,
            }
            if scores is not None:
                result["scores"] = torch.stack(scores)
            return result
        else:
            return generated_ids