import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CausalSelfAttention(nn.Module):
    """Single model causal self-attention - we'll replicate this across models."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3 * self.d_k).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn_weights) @ v

        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(attn)


class TransformerBlock(nn.Module):
    """Single model transformer block - we'll replicate this across models."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerModels(nn.Module):
    """
    Multi-model decoder-only transformer for counting sequences.
    Uses parameter replication approach similar to LeNetModels.
    """
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, model_count, device='cuda'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.model_count = model_count
        self.device = device
        
        # Create parameters for model_count independent transformers
        # Token embeddings - replicated for each model
        self.token_emb = nn.Parameter(torch.randn(model_count, vocab_size, d_model))
        
        # Position embeddings - replicated for each model
        self.pos_emb = nn.Parameter(torch.randn(model_count, max_len, d_model))
        
        # Transformer blocks parameters - we'll store all parameters directly
        self.transformer_params = nn.ParameterDict()
        
        # Initialize parameters for each layer
        for layer in range(n_layers):
            # Layer norm 1
            self.transformer_params[f'ln1_{layer}_weight'] = nn.Parameter(torch.ones(model_count, d_model))
            self.transformer_params[f'ln1_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
            
            # Attention QKV (weight shape: out_features x in_features for F.linear)
            self.transformer_params[f'attn_{layer}_qkv_weight'] = nn.Parameter(torch.randn(model_count, 3*d_model, d_model))
            self.transformer_params[f'attn_{layer}_qkv_bias'] = nn.Parameter(torch.zeros(model_count, 3*d_model))
            
            # Attention output
            self.transformer_params[f'attn_{layer}_out_weight'] = nn.Parameter(torch.randn(model_count, d_model, d_model))
            self.transformer_params[f'attn_{layer}_out_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
            
            # Layer norm 2
            self.transformer_params[f'ln2_{layer}_weight'] = nn.Parameter(torch.ones(model_count, d_model))
            self.transformer_params[f'ln2_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
            
            # Feed forward (weight shape: out_features x in_features for F.linear)
            self.transformer_params[f'ff1_{layer}_weight'] = nn.Parameter(torch.randn(model_count, d_ff, d_model))
            self.transformer_params[f'ff1_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_ff))
            self.transformer_params[f'ff2_{layer}_weight'] = nn.Parameter(torch.randn(model_count, d_model, d_ff))
            self.transformer_params[f'ff2_{layer}_bias'] = nn.Parameter(torch.zeros(model_count, d_model))
        
        # Final layer norm
        self.ln_f_weight = nn.Parameter(torch.ones(model_count, d_model))
        self.ln_f_bias = nn.Parameter(torch.zeros(model_count, d_model))
        
        # Output head (weight shape: out_features x in_features for F.linear)
        self.head_weight = nn.Parameter(torch.randn(model_count, vocab_size, d_model))
        self.head_bias = nn.Parameter(torch.zeros(model_count, vocab_size))
        
        # Pattern search state
        self.basis_list = None
        self.curr_idx = 0
        self.radius = 1.0
        
        # Initialize parameters properly
        self._init_multi_model_params()
        
    def _init_multi_model_params(self):
        """Initialize parameters for multi-model setup."""
        # Initialize all models with same weights initially
        with torch.no_grad():
            # Token and position embeddings
            nn.init.normal_(self.token_emb, std=0.02)
            nn.init.normal_(self.pos_emb, std=0.02)
            
            # Make all models start with same weights
            for i in range(1, self.model_count):
                self.token_emb[i] = self.token_emb[0].clone()
                self.pos_emb[i] = self.pos_emb[0].clone()
            
            # Initialize transformer parameters
            for name, param in self.transformer_params.items():
                if 'weight' in name and 'ln' not in name:
                    # Xavier initialization for linear layers
                    for i in range(self.model_count):
                        if len(param.shape) == 3:  # weight matrices
                            nn.init.xavier_uniform_(param[i])
                        else:  # biases
                            nn.init.zeros_(param[i])
                elif 'ln' in name and 'weight' in name:
                    # Layer norm weights to 1
                    nn.init.ones_(param)
                elif 'ln' in name and 'bias' in name:
                    # Layer norm biases to 0
                    nn.init.zeros_(param)
                
                # Make all models start with same weights
                if param.dim() > 1:
                    for i in range(1, self.model_count):
                        param[i] = param[0].clone()
            
            # Initialize head
            for i in range(self.model_count):
                nn.init.xavier_uniform_(self.head_weight[i])
                nn.init.zeros_(self.head_bias[i])
            
            # Make all models start with same head weights
            for i in range(1, self.model_count):
                self.head_weight[i] = self.head_weight[0].clone()
                self.head_bias[i] = self.head_bias[0].clone()
    
    def _attention_forward(self, x, layer_idx, model_idx):
        """Forward pass for attention layer of specific model."""
        B, T, C = x.shape
        
        # QKV projection
        qkv_weight = self.transformer_params[f'attn_{layer_idx}_qkv_weight'][model_idx]  # (3*d_model, d_model)
        qkv_bias = self.transformer_params[f'attn_{layer_idx}_qkv_bias'][model_idx]    # (3*d_model,)
        qkv = F.linear(x, qkv_weight, qkv_bias)  # (B, T, 3*d_model)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_model // self.n_heads).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, n_heads, T, d_k)
        
        # Compute attention
        d_k = self.d_model // self.n_heads
        attn_scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = attn_weights @ v  # (B, n_heads, T, d_k)
        
        # Reshape and project
        attn = attn.transpose(1, 2).contiguous().reshape(B, T, C)  # (B, T, d_model)
        
        # Output projection
        out_weight = self.transformer_params[f'attn_{layer_idx}_out_weight'][model_idx]  # (d_model, d_model)
        out_bias = self.transformer_params[f'attn_{layer_idx}_out_bias'][model_idx]      # (d_model,)
        return F.linear(attn, out_weight, out_bias)
    
    def forward(self, x, position_ids=None):
        """
        Forward pass for multiple models.
        
        Args:
            x: Input tokens (batch_size, seq_len)
            position_ids: Position indices for length generalization (batch_size, seq_len)
            
        Returns:
            Logits: (batch_size, model_count, seq_len, vocab_size)
        """
        B, T = x.size()
        
        # Process each model independently
        all_logits = []
        
        for model_idx in range(self.model_count):
            # Token embeddings
            token_emb = F.embedding(x, self.token_emb[model_idx])  # (B, T, d_model)
            
            # Position embeddings
            if position_ids is not None:
                # Use custom position indices
                pos_emb = self.pos_emb[model_idx][position_ids]  # (B, T, d_model)
            else:
                # Standard sequential positions
                pos_emb = self.pos_emb[model_idx][:T].unsqueeze(0).expand(B, -1, -1)  # (B, T, d_model)
            
            # Add embeddings
            hidden = token_emb + pos_emb  # (B, T, d_model)
            
            # Pass through transformer blocks
            for layer_idx in range(self.n_layers):
                # Layer norm 1
                ln1_weight = self.transformer_params[f'ln1_{layer_idx}_weight'][model_idx]
                ln1_bias = self.transformer_params[f'ln1_{layer_idx}_bias'][model_idx]
                normed = F.layer_norm(hidden, (self.d_model,), ln1_weight, ln1_bias)
                
                # Attention
                attn_out = self._attention_forward(normed, layer_idx, model_idx)
                hidden = hidden + attn_out  # Residual connection
                
                # Layer norm 2
                ln2_weight = self.transformer_params[f'ln2_{layer_idx}_weight'][model_idx]
                ln2_bias = self.transformer_params[f'ln2_{layer_idx}_bias'][model_idx]
                normed = F.layer_norm(hidden, (self.d_model,), ln2_weight, ln2_bias)
                
                # Feed forward
                ff1_weight = self.transformer_params[f'ff1_{layer_idx}_weight'][model_idx]
                ff1_bias = self.transformer_params[f'ff1_{layer_idx}_bias'][model_idx]
                ff_out = F.linear(normed, ff1_weight, ff1_bias)
                ff_out = F.relu(ff_out)
                
                ff2_weight = self.transformer_params[f'ff2_{layer_idx}_weight'][model_idx]
                ff2_bias = self.transformer_params[f'ff2_{layer_idx}_bias'][model_idx]
                ff_out = F.linear(ff_out, ff2_weight, ff2_bias)
                
                hidden = hidden + ff_out  # Residual connection
            
            # Final layer norm
            hidden = F.layer_norm(hidden, (self.d_model,), self.ln_f_weight[model_idx], self.ln_f_bias[model_idx])
            
            # Output head
            logits = F.linear(hidden, self.head_weight[model_idx], self.head_bias[model_idx])  # (B, T, vocab_size)
            all_logits.append(logits)
        
        # Stack all models
        all_logits = torch.stack(all_logits, dim=1)  # (B, model_count, T, vocab_size)
        return all_logits

    @torch.no_grad()
    def pattern_search(self, x, y, loss_func):
        """
        Pattern search optimization similar to LeNetModels.
        Tests different parameter perturbations across multiple models.
        """
        if self.basis_list is None:
            self.basis_list = []
            for para in self.parameters():
                para_flatten = para.data.view(self.model_count, -1)
                for p in range(para_flatten.shape[1]):
                    self.basis_list.append((para_flatten, p, "+"))
                    self.basis_list.append((para_flatten, p, "-"))
        
        random.shuffle(self.basis_list)
        self.curr_idx = 0

        while True:
            # Replicate the first model across all model copies
            for para in self.parameters():
                original_shape = para.shape
                para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                para_reshaped[1:] = para_reshaped[0:1]

            # Modify each model at one parameter location
            for i in range(1, self.model_count):
                if self.curr_idx >= len(self.basis_list):
                    print("Pattern search: went over all parameters")
                    random.shuffle(self.basis_list)
                    self.radius /= 2
                    self.curr_idx = 0
                    break
                
                para, p_i, op = self.basis_list[self.curr_idx]
                if op == "+":
                    para[i, p_i] += self.radius
                else:
                    para[i, p_i] -= self.radius
                self.curr_idx += 1
                
                if self.curr_idx >= len(self.basis_list):
                    print("Pattern search: completed parameter sweep")
                    random.shuffle(self.basis_list)
                    self.radius /= 2
                    self.curr_idx = 0
                    break

            # Forward pass and select best model
            pred = self.forward_normalize(x)  # (batch_size, model_count, seq_len, vocab_size)
            n, m, t, o = pred.shape
            
            # Compute loss for each model
            loss = loss_func(
                pred.reshape(n * m * t, o), 
                y.repeat(1, m).view(-1)
            ).view(n, m, t).mean(dim=(0, 2))  # Average over batch and sequence
            
            best_idx = loss.min(dim=0).indices

            # Copy best model to position 0
            for para in self.parameters():
                original_shape = para.shape
                para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                para_reshaped[:] = para_reshaped[best_idx:best_idx+1]
            
            if best_idx != 0:
                break

    @torch.no_grad() 
    def greedy_random(self, x, y, loss_func):
        """
        Greedy random search optimization similar to LeNetModels.
        """
        for _ in range(30):
            iter_max = 100
            for i in range(iter_max):
                # Add noise to all models except the first one
                for para in self.parameters():
                    original_shape = para.shape
                    para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                    para_reshaped[1:] = para_reshaped[0:1]
                    para_reshaped[1:] += torch.randn_like(para_reshaped[1:]) * self.radius

                # Forward pass and select best model
                pred = self.forward_normalize(x)
                n, m, t, o = pred.shape
                
                loss = loss_func(
                    pred.reshape(n * m * t, o), 
                    y.repeat_interleave(m * t)
                ).view(n, m, t).mean(dim=(0, 2))

                best_idx = loss.min(dim=0).indices

                # Copy best model to all positions
                for para in self.parameters():
                    original_shape = para.shape
                    para_reshaped = para.data.view(self.model_count, -1, *original_shape[2:])
                    para_reshaped[:] = para_reshaped[best_idx:best_idx + 1]
                
                if best_idx != 0:
                    return
                    
            print(f"Greedy random: radius decreased to {self.radius/2}")
            self.radius /= 2

    @torch.no_grad()
    def get_model_subsets(self, idx):
        """Get a subset of models by index."""
        model_count = len(idx)
        new_model = TransformerModels(
            vocab_size=self.vocab_size,
            d_model=self.d_model, 
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            max_len=self.max_len,
            model_count=model_count,
            device=self.device
        )
        new_model.load_state_dict(self.get_weights_by_idx(idx))
        return new_model

    @torch.no_grad()
    def get_weights_by_idx(self, idx):
        """Extract weights for specific model indices."""
        weight_dict = {}
        for name, para in self.state_dict().items():
            original_shape = para.shape
            if 'token_emb' in name or 'pos_emb' in name:
                # These are shared, just slice the expanded dimension
                para_reshaped = para.reshape(self.model_count, -1, *original_shape[2:])
                para_selected = para_reshaped[idx]
                para_selected = para_selected.reshape(-1, *original_shape[1:])
            else:
                # These are per-model parameters
                para_reshaped = para.reshape(self.model_count, -1, *original_shape[2:])
                para_selected = para_reshaped[idx] 
                para_selected = para_selected.reshape(-1, *original_shape[1:])
            weight_dict[name] = para_selected.clone().detach().cpu()
        return weight_dict

    @torch.no_grad()
    def reinitialize(self, mult=1):
        """Reinitialize all parameters."""
        for para in self.parameters():
            torch.nn.init.uniform_(para.data, a=-mult, b=mult)

    @torch.no_grad()
    def reset_parameters(self):
        """Reset parameters using standard initialization."""
        self._init_multi_model_params()

    def forward_normalize(self, x, position_ids=None):
        """Forward pass with parameter normalization (for consistent comparison)."""
        return self.forward(x, position_ids)
        
    @torch.no_grad()
    def shorten(self, count):
        """Return model with fewer copies."""
        idx = torch.arange(count)
        return self.get_model_subsets(idx)