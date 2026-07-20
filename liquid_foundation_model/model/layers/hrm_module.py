import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HighLevelModule(nn.Module):
    """
    High-level module for abstract planning and slow reasoning.
    Operates at a slower timescale for strategic thinking.
    """
    
    def __init__(self, hidden_size: int, planning_size: int = None, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.planning_size = planning_size or hidden_size // 2
        
        # Planning state
        self.planning_rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=self.planning_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Abstract reasoning layers
        self.abstract_proj = nn.Linear(self.planning_size, hidden_size)
        self.planning_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, planning_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass for high-level planning.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            planning_state: Previous planning state (h, c)
            
        Returns:
            planning_output: Abstract planning signal
            new_planning_state: Updated planning state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize planning state if not provided
        if planning_state is None:
            h0 = torch.zeros(self.planning_rnn.num_layers, batch_size, self.planning_size, 
                           device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.planning_rnn.num_layers, batch_size, self.planning_size,
                           device=x.device, dtype=x.dtype)
            planning_state = (h0, c0)
        
        # Process through planning RNN (operates at slower timescale)
        # We can subsample or process every few tokens for efficiency
        planning_input = x.mean(dim=1, keepdim=True)  # Aggregate sequence for planning
        planning_output, new_planning_state = self.planning_rnn(planning_input, planning_state)
        
        # Project planning output back to hidden size
        abstract_signal = self.abstract_proj(planning_output)
        
        # Broadcast planning signal across sequence
        abstract_signal = abstract_signal.expand(-1, seq_len, -1)
        
        # Gate the planning signal with input
        combined = torch.cat([x, abstract_signal], dim=-1)
        gated_output = torch.sigmoid(self.planning_gate(combined)) * abstract_signal
        
        # Add residual connection and normalize
        output = self.norm(x + gated_output)
        
        return output, new_planning_state


class LowLevelModule(nn.Module):
    """
    Low-level module for rapid, detailed computations.
    Operates at a faster timescale for immediate processing.
    """
    
    def __init__(self, hidden_size: int, detail_size: int = None, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.detail_size = detail_size or hidden_size
        
        # Fast processing RNN
        self.detail_rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=self.detail_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Detail processing layers
        self.detail_proj = nn.Linear(self.detail_size, hidden_size)
        self.detail_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, planning_signal: torch.Tensor, 
                detail_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for low-level detailed processing.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            planning_signal: High-level planning guidance
            detail_state: Previous detail state
            
        Returns:
            detail_output: Detailed processing result
            new_detail_state: Updated detail state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize detail state if not provided
        if detail_state is None:
            detail_state = torch.zeros(self.detail_rnn.num_layers, batch_size, self.detail_size,
                                     device=x.device, dtype=x.dtype)
        
        # Combine input with planning guidance
        guided_input = x + planning_signal
        
        # Process through detail RNN (operates at faster timescale)
        detail_output, new_detail_state = self.detail_rnn(guided_input, detail_state)
        
        # Project detail output
        detail_features = self.detail_proj(detail_output)
        
        # Gate detail features with original input
        combined = torch.cat([x, detail_features], dim=-1)
        gated_output = torch.sigmoid(self.detail_gate(combined)) * detail_features
        
        # Add residual connection and normalize
        output = self.norm(x + gated_output)
        
        return output, new_detail_state


class HierarchicalReasoningModule(nn.Module):
    """
    Complete HRM module combining high-level planning and low-level execution.
    """
    
    def __init__(self, hidden_size: int, planning_size: int = None, detail_size: int = None,
                 num_reasoning_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_reasoning_steps = num_reasoning_steps
        
        # High-level and low-level modules
        self.high_level = HighLevelModule(hidden_size, planning_size)
        self.low_level = LowLevelModule(hidden_size, detail_size)
        
        # Cross-module communication
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, use_reasoning: bool = True) -> torch.Tensor:
        """
        Forward pass through the hierarchical reasoning module.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            use_reasoning: Whether to use multi-step reasoning
            
        Returns:
            output: Processed tensor with hierarchical reasoning
        """
        if not use_reasoning:
            return x
        
        batch_size, seq_len, hidden_size = x.shape
        
        # Initialize states
        planning_state = None
        detail_state = None
        current_input = x
        
        # Multi-step hierarchical reasoning
        for step in range(self.num_reasoning_steps):
            # High-level planning
            planning_output, planning_state = self.high_level(current_input, planning_state)
            
            # Low-level detailed processing guided by planning
            detail_output, detail_state = self.low_level(current_input, planning_output, detail_state)
            
            # Cross-module attention for information integration
            attended_output, _ = self.cross_attention(
                query=detail_output,
                key=planning_output,
                value=planning_output
            )
            
            # Combine outputs
            current_input = detail_output + attended_output
        
        # Final output projection
        output = self.output_proj(current_input)
        output = self.output_norm(output + x)  # Residual connection with original input
        
        return output


class TaskRouter(nn.Module):
    """
    Routes inputs to appropriate processing based on task complexity.
    """
    
    def __init__(self, hidden_size: int, num_task_types: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_task_types = num_task_types
        
        # Task classification
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_task_types)
        )
        
        # Task type embeddings
        self.task_embeddings = nn.Embedding(num_task_types, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify task type and return routing decision.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            task_logits: Task classification logits
            task_embedding: Task-specific embedding
        """
        # Use mean pooling for sequence-level classification
        pooled = x.mean(dim=1)  # [batch_size, hidden_size]
        
        # Classify task type
        task_logits = self.task_classifier(pooled)  # [batch_size, num_task_types]
        
        # Get task embedding
        task_type = torch.argmax(task_logits, dim=-1)  # [batch_size]
        task_embedding = self.task_embeddings(task_type)  # [batch_size, hidden_size]
        
        return task_logits, task_embedding.unsqueeze(1)  # Add seq_len dimension