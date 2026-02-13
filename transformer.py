"""
=============================================================================
Transformer Neural Network for Character-Level Machine Translation
=============================================================================
This module implements the full Transformer architecture (Vaswani et al., 
"Attention Is All You Need", 2017) for translating English sentences to 
Kannada at the character level.

Architecture Overview:
    English sentence → [Encoder] → context embeddings
                                        ↓
    Kannada sentence → [Decoder] → predicted next characters → [Linear] → logits

Key design choices:
  - Character-level tokenization (not word or subword/BPE)
  - Separate character-to-index mappings for English and Kannada
  - Special tokens: START, END, PADDING
  - Autoregressive decoding at inference (one character at a time)
  - Parallelized training with teacher forcing
=============================================================================
"""

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: Device Selection
# ---------------------------------------------------------------------------
def get_device():
    """Return CUDA device if available, otherwise CPU."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
def scaled_dot_product(q, k, v, mask=None):
    """
    Core attention mechanism used by all attention layers.

    Computes:  Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V

    Args:
        q: Query tensor  — shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor    — shape (batch, num_heads, seq_len, head_dim)
        v: Value tensor  — shape (batch, num_heads, seq_len, head_dim)
        mask: Optional mask to block certain positions (padding / look-ahead).
              Masked positions are set to -inf so softmax drives them to ~0.

    Returns:
        values:    Weighted sum of V — shape (batch, num_heads, seq_len, head_dim)
        attention: Attention weights  — shape (batch, num_heads, seq_len, seq_len)
    """
    d_k = q.size()[-1]  # Dimension of each attention head (head_dim)

    # Dot product between queries and keys, scaled by √d_k to prevent
    # large dot-product magnitudes that would push softmax into saturated regions
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    # Apply mask (e.g., padding mask or causal look-ahead mask).
    # The mask contains 0 for valid positions and -inf for blocked positions.
    # permute is needed to broadcast the mask across the batch dimension correctly.
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask  # Add mask (broadcast over batch)
        scaled = scaled.permute(1, 0, 2, 3)          # Restore original dimension order

    # Softmax along the last dim (key positions) to get attention weights summing to 1
    attention = F.softmax(scaled, dim=-1)

    # Weighted combination of value vectors using the attention weights
    values = torch.matmul(attention, v)
    return values, attention


# ===========================================================================
# POSITIONAL ENCODING
# ===========================================================================
class PositionalEncoding(nn.Module):
    """
    Injects positional information into embeddings since the Transformer
    has no built-in notion of token order (unlike RNNs).

    Uses sinusoidal functions of different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This allows the model to learn relative positions because for any fixed
    offset k, PE(pos+k) can be represented as a linear function of PE(pos).
    """
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length  # Max number of characters in a sentence
        self.d_model = d_model                          # Embedding dimension (e.g., 512)

    def forward(self):
        # Even indices: 0, 2, 4, ... used to compute the frequency denominator
        even_i = torch.arange(0, self.d_model, 2).float()
        # Denominator = 10000^(2i / d_model) — higher dimensions get lower frequencies
        denominator = torch.pow(10000, even_i/self.d_model)
        # Position indices: 0, 1, 2, ..., max_seq_len-1 as a column vector
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        # Compute sin for even indices, cos for odd indices
        even_PE = torch.sin(position / denominator)  # shape: (max_seq_len, d_model/2)
        odd_PE = torch.cos(position / denominator)   # shape: (max_seq_len, d_model/2)
        # Interleave sin and cos values: [sin_0, cos_0, sin_1, cos_1, ...]
        stacked = torch.stack([even_PE, odd_PE], dim=2)          # (max_seq_len, d_model/2, 2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)      # (max_seq_len, d_model)
        return PE


# ===========================================================================
# SENTENCE EMBEDDING (Tokenization + Word Embedding + Positional Encoding)
# ===========================================================================
class SentenceEmbedding(nn.Module):
    """
    Converts a raw sentence (string of characters) into a sequence of
    d_model-dimensional vectors that the Transformer can process.

    Pipeline:  raw chars → integer indices → embedding vectors → + positional encoding → dropout

    This is used by both the Encoder (for English) and Decoder (for Kannada),
    each with their own character-to-index mapping.
    """
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)          # Number of unique characters in the language
        self.max_sequence_length = max_sequence_length    # Fixed length all sequences are padded/truncated to
        self.embedding = nn.Embedding(self.vocab_size, d_model)  # Learnable lookup table: index → d_model vector
        self.language_to_index = language_to_index        # Dict mapping each character → unique integer index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)                  # Regularization to prevent overfitting
        self.START_TOKEN = START_TOKEN    # Special token marking the start of a sentence
        self.END_TOKEN = END_TOKEN        # Special token marking the end of a sentence
        self.PADDING_TOKEN = PADDING_TOKEN  # Special token for padding shorter sentences
    
    def batch_tokenize(self, batch, start_token, end_token):
        """
        Convert a batch of raw sentences (list of strings) into a padded
        tensor of integer indices.

        Args:
            batch: List of sentence strings
            start_token: Whether to prepend the START token
            end_token: Whether to append the END token

        Returns:
            Tensor of shape (batch_size, max_sequence_length) with integer indices
        """

        def tokenize(sentence, start_token, end_token):
            """Convert a single sentence string into a list of token indices."""
            # Map each character to its integer index
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            # Optionally prepend START token (used for decoder input during training)
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            # Optionally append END token (signals end of sentence)
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            # Pad the sequence to max_sequence_length with PADDING tokens
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        # Tokenize every sentence in the batch
        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)  # Stack into (batch_size, max_seq_len) tensor
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token):
        """
        Full embedding pipeline: tokenize → embed → add positional encoding → dropout.

        Args:
            x: List of raw sentence strings
            start_token: Whether to add START token
            end_token: Whether to add END token

        Returns:
            Tensor of shape (batch_size, max_seq_len, d_model) — contextualized embeddings
        """
        x = self.batch_tokenize(x, start_token, end_token)  # (batch, max_seq_len) integers
        x = self.embedding(x)              # (batch, max_seq_len, d_model) learned embeddings
        pos = self.position_encoder().to(get_device())  # (max_seq_len, d_model) positional signal
        x = self.dropout(x + pos)          # Add position info and apply dropout
        return x


# ===========================================================================
# MULTI-HEAD SELF-ATTENTION
# ===========================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention: allows the model to jointly attend to
    information from different representation subspaces at different positions.

    Instead of computing a single attention function, the input is projected
    into num_heads separate (Q, K, V) sets, attention is computed in parallel
    for each head, and the results are concatenated and linearly projected.

    Used in:
      - Encoder self-attention (each English char attends to all English chars)
      - Decoder masked self-attention (each Kannada char attends to previous Kannada chars only)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model              # Total embedding dimension (e.g., 512)
        self.num_heads = num_heads          # Number of parallel attention heads (e.g., 8)
        self.head_dim = d_model // num_heads  # Dimension per head (e.g., 64)
        # Single linear layer that produces Q, K, V all at once (more efficient than 3 separate layers)
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        # Final linear projection after concatenating all heads
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        """
        Args:
            x: Input tensor — shape (batch, seq_len, d_model)
            mask: Attention mask to block padding/future tokens

        Returns:
            Output tensor — shape (batch, seq_len, d_model)
        """
        batch_size, sequence_length, d_model = x.size()

        # Project input into Q, K, V concatenated — shape: (batch, seq_len, 3*d_model)
        qkv = self.qkv_layer(x)

        # Reshape to separate the heads — (batch, seq_len, num_heads, 3*head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)

        # Reorder to (batch, num_heads, seq_len, 3*head_dim) so each head is processed independently
        qkv = qkv.permute(0, 2, 1, 3)

        # Split the last dimension into Q, K, V — each is (batch, num_heads, seq_len, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute scaled dot-product attention for all heads in parallel
        values, attention = scaled_dot_product(q, k, v, mask)

        # Concatenate heads: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        # Final linear projection to mix information across heads
        out = self.linear_layer(values)
        return out


# ===========================================================================
# LAYER NORMALIZATION
# ===========================================================================
class LayerNormalization(nn.Module):
    """
    Layer Normalization: normalizes inputs across the feature dimension
    (d_model) for each token independently.

    Formula:  y = gamma * (x - mean) / std + beta

    Unlike Batch Normalization, Layer Norm:
      - Normalizes across features (not across the batch)
      - Works well with variable-length sequences
      - Is independent of batch size

    gamma (scale) and beta (shift) are learnable parameters that allow the
    model to undo the normalization if needed.
    """
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape  # Shape of the feature dimensions to normalize over
        self.eps = eps                             # Small constant to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(parameters_shape))   # Learnable scale (initialized to 1)
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))  # Learnable shift (initialized to 0)

    def forward(self, inputs):
        # Compute which dimensions to normalize over (last N dims based on parameters_shape)
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]  # e.g., [-1] for [d_model]
        mean = inputs.mean(dim=dims, keepdim=True)                    # Mean across feature dims
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)     # Variance
        std = (var + self.eps).sqrt()                                  # Standard deviation (with eps for stability)
        y = (inputs - mean) / std                                      # Normalize to zero mean, unit variance
        out = self.gamma * y + self.beta                               # Scale and shift
        return out


# ===========================================================================
# POSITION-WISE FEED-FORWARD NETWORK (FFN)
# ===========================================================================
class PositionwiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network applied independently to each position.

    FFN(x) = Linear2( Dropout( ReLU( Linear1(x) ) ) )

    This is applied to every token's representation separately and identically.
    The hidden layer typically has a larger dimension (e.g., 2048) than d_model (e.g., 512),
    allowing the network to learn richer intermediate representations.
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)   # Expand: d_model → hidden (e.g., 512 → 2048)
        self.linear2 = nn.Linear(hidden, d_model)    # Contract: hidden → d_model (e.g., 2048 → 512)
        self.relu = nn.ReLU()                         # Non-linear activation
        self.dropout = nn.Dropout(p=drop_prob)        # Regularization

    def forward(self, x):
        x = self.linear1(x)   # Project up to hidden dimension
        x = self.relu(x)      # Apply non-linearity
        x = self.dropout(x)   # Dropout for regularization
        x = self.linear2(x)   # Project back down to d_model
        return x


# ===========================================================================
# ENCODER LAYER (Single Layer of the Encoder Stack)
# ===========================================================================
class EncoderLayer(nn.Module):
    """
    One layer of the Transformer Encoder. The full encoder stacks N of these.

    Each layer has two sub-layers:
      1. Multi-Head Self-Attention  → lets each character attend to all other characters
      2. Position-wise Feed-Forward → applies a non-linear transformation to each position

    Both sub-layers use:
      - Residual connections (x + sublayer(x)) to help gradient flow
      - Layer Normalization for training stability
      - Dropout for regularization
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        # Sub-layer 1: Multi-Head Self-Attention
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        # Sub-layer 2: Position-wise Feed-Forward Network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        """
        Args:
            x: Input tensor — (batch, seq_len, d_model)
            self_attention_mask: Mask to block padding tokens in self-attention

        Returns:
            Output tensor — (batch, seq_len, d_model)
        """
        # --- Sub-layer 1: Self-Attention with residual connection & layer norm ---
        residual_x = x.clone()                        # Save input for residual connection
        x = self.attention(x, mask=self_attention_mask)  # Multi-head self-attention
        x = self.dropout1(x)                           # Dropout
        x = self.norm1(x + residual_x)                 # Add residual & normalize

        # --- Sub-layer 2: Feed-Forward with residual connection & layer norm ---
        residual_x = x.clone()                         # Save input for residual connection
        x = self.ffn(x)                                # Feed-forward network
        x = self.dropout2(x)                           # Dropout
        x = self.norm2(x + residual_x)                 # Add residual & normalize
        return x


# ===========================================================================
# SEQUENTIAL ENCODER (Helper to stack multiple EncoderLayers)
# ===========================================================================
class SequentialEncoder(nn.Sequential):
    """
    Custom Sequential container that passes both the input tensor AND the
    attention mask through each encoder layer. The standard nn.Sequential
    only passes a single tensor, so we override forward().
    """
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)  # Each layer receives the mask
        return x


# ===========================================================================
# FULL ENCODER
# ===========================================================================
class Encoder(nn.Module):
    """
    The complete Transformer Encoder.

    Pipeline:
      English sentence (string) → SentenceEmbedding → stack of N EncoderLayers → output

    The output is a sequence of context-aware character embeddings that capture
    the meaning of each English character in the context of the full sentence.
    These embeddings are passed to the Decoder via cross-attention.
    """
    def __init__(self, 
                 d_model,              # Embedding dimension (e.g., 512)
                 ffn_hidden,           # FFN hidden layer size (e.g., 2048)
                 num_heads,            # Number of attention heads (e.g., 8)
                 drop_prob,            # Dropout probability (e.g., 0.1)
                 num_layers,           # Number of encoder layers to stack (e.g., 5)
                 max_sequence_length,  # Max character sequence length (e.g., 200)
                 language_to_index,    # English character → index mapping dict
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        # Embedding layer: converts raw English characters to d_model vectors with position info
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        # Stack of N identical encoder layers
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        """
        Args:
            x: Batch of raw English sentence strings
            self_attention_mask: Mask to block padding tokens
            start_token: Whether to prepend START token
            end_token: Whether to append END token

        Returns:
            Context-aware embeddings — (batch, seq_len, d_model)
        """
        x = self.sentence_embedding(x, start_token, end_token)  # Tokenize + embed + positional encode
        x = self.layers(x, self_attention_mask)                   # Pass through all encoder layers
        return x


# ===========================================================================
# MULTI-HEAD CROSS-ATTENTION (Decoder attends to Encoder output)
# ===========================================================================
class MultiHeadCrossAttention(nn.Module):
    """
    Cross-Attention: allows the Decoder to attend to the Encoder's output.

    Unlike self-attention where Q, K, V all come from the same sequence:
      - Keys (K) and Values (V) come from the ENCODER output (English context)
      - Queries (Q) come from the DECODER (Kannada sequence being generated)

    This is how the decoder "looks at" the source English sentence to decide
    what Kannada character to produce next.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # K and V are derived from encoder output (source language)
        self.kv_layer = nn.Linear(d_model , 2 * d_model)  # Projects encoder output → K, V
        # Q is derived from decoder's current state (target language)
        self.q_layer = nn.Linear(d_model , d_model)        # Projects decoder state → Q
        self.linear_layer = nn.Linear(d_model, d_model)    # Final output projection
    
    def forward(self, x, y, mask):
        """
        Args:
            x: Encoder output — (batch, seq_len, d_model) — source of K, V
            y: Decoder state  — (batch, seq_len, d_model) — source of Q
            mask: Cross-attention mask (masks padding in encoder output)

        Returns:
            Output tensor — (batch, seq_len, d_model)
        """
        batch_size, sequence_length, d_model = x.size()

        # Generate K, V from encoder output (English context)
        kv = self.kv_layer(x)  # (batch, seq_len, 2*d_model)
        # Generate Q from decoder state (Kannada context so far)
        q = self.q_layer(y)    # (batch, seq_len, d_model)

        # Reshape to separate attention heads
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Reorder to (batch, num_heads, seq_len, dim) for parallel head computation
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)

        # Split K and V from the combined kv tensor
        k, v = kv.chunk(2, dim=-1)  # Each: (batch, num_heads, seq_len, head_dim)

        # Compute cross-attention (decoder queries attend to encoder keys/values)
        values, attention = scaled_dot_product(q, k, v, mask)

        # Concatenate heads and project back to d_model
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


# ===========================================================================
# DECODER LAYER (Single Layer of the Decoder Stack)
# ===========================================================================
class DecoderLayer(nn.Module):
    """
    One layer of the Transformer Decoder. The full decoder stacks N of these.

    Each layer has three sub-layers:
      1. Masked Multi-Head Self-Attention → Kannada chars attend to previous Kannada chars only
         (look-ahead mask prevents attending to future tokens during training)
      2. Multi-Head Cross-Attention → Kannada chars attend to all English encoder outputs
      3. Position-wise Feed-Forward → non-linear transformation at each position

    All sub-layers use residual connections + layer norm + dropout.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        # Sub-layer 1: Masked Self-Attention (Kannada → Kannada, with look-ahead mask)
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Sub-layer 2: Cross-Attention (Kannada queries → English keys/values)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        # Sub-layer 3: Feed-Forward Network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        """
        Args:
            x: Encoder output           — (batch, seq_len, d_model) — English context
            y: Decoder input             — (batch, seq_len, d_model) — Kannada so far
            self_attention_mask: Masks padding + future tokens (look-ahead) in decoder self-attention
            cross_attention_mask: Masks padding tokens when decoder attends to encoder output

        Returns:
            Updated decoder state — (batch, seq_len, d_model)
        """
        # --- Sub-layer 1: Masked Self-Attention (Kannada attending to itself) ---
        _y = y.clone()                                         # Save for residual
        y = self.self_attention(y, mask=self_attention_mask)    # Self-attend with look-ahead mask
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)                           # Add residual & normalize

        # --- Sub-layer 2: Cross-Attention (Kannada attending to English) ---
        _y = y.clone()                                                    # Save for residual
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)  # Cross-attend to encoder
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)                                      # Add residual & normalize

        # --- Sub-layer 3: Feed-Forward Network ---
        _y = y.clone()                   # Save for residual
        y = self.ffn(y)                  # Feed-forward
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)     # Add residual & normalize
        return y


# ===========================================================================
# SEQUENTIAL DECODER (Helper to stack multiple DecoderLayers)
# ===========================================================================
class SequentialDecoder(nn.Sequential):
    """
    Custom Sequential container for decoder layers.
    Passes encoder output (x), decoder state (y), and both masks through
    each decoder layer. Only the decoder state (y) is updated between layers;
    the encoder output (x) stays the same throughout.
    """
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


# ===========================================================================
# FULL DECODER
# ===========================================================================
class Decoder(nn.Module):
    """
    The complete Transformer Decoder.

    Pipeline:
      Kannada sentence → SentenceEmbedding → stack of N DecoderLayers → output

    During training:
      - Receives the target Kannada sentence (with START token, shifted right)
      - Uses teacher forcing: all positions computed in parallel
      - Look-ahead mask prevents attending to future characters

    During inference (autoregressive):
      - Starts with just the START token
      - Generates one character at a time, feeding each prediction back as input
      - Stops when END token is generated
    """
    def __init__(self, 
                 d_model,              # Embedding dimension
                 ffn_hidden,           # FFN hidden layer size
                 num_heads,            # Number of attention heads
                 drop_prob,            # Dropout probability
                 num_layers,           # Number of decoder layers to stack
                 max_sequence_length,  # Max character sequence length
                 language_to_index,    # Kannada character → index mapping dict
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        # Embedding layer: converts raw Kannada characters to d_model vectors with position info
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        # Stack of N identical decoder layers
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        """
        Args:
            x: Encoder output        — (batch, seq_len, d_model) — from the Encoder
            y: Batch of raw Kannada sentence strings
            self_attention_mask: Look-ahead + padding mask for decoder self-attention
            cross_attention_mask: Padding mask for cross-attention
            start_token: Whether to prepend START token to Kannada input
            end_token: Whether to append END token to Kannada input

        Returns:
            Decoder output — (batch, seq_len, d_model)
        """
        y = self.sentence_embedding(y, start_token, end_token)  # Tokenize + embed Kannada chars
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)  # Pass through all decoder layers
        return y


# ===========================================================================
# FULL TRANSFORMER MODEL
# ===========================================================================
class Transformer(nn.Module):
    """
    Complete Transformer model for English → Kannada character-level translation.

    Architecture:
      1. Encoder: Processes English input → produces context-aware embeddings
      2. Decoder: Takes Kannada input + encoder output → produces output representations
      3. Linear:  Projects decoder output to Kannada vocabulary size for character prediction

    The final linear layer outputs logits (unnormalized scores) for each character
    in the Kannada vocabulary. During training, these are passed to CrossEntropyLoss.
    During inference, softmax is applied to get probabilities for the next character.

    Masks:
      - encoder_self_attention_mask: Blocks padding tokens in English input
      - decoder_self_attention_mask: Blocks padding + future tokens in Kannada input (causal/look-ahead mask)
      - decoder_cross_attention_mask: Blocks padding tokens when decoder attends to encoder output
    """
    def __init__(self, 
                d_model,              # Embedding dimension (e.g., 512)
                ffn_hidden,           # FFN hidden size (e.g., 2048)
                num_heads,            # Number of attention heads (e.g., 8)
                drop_prob,            # Dropout probability (e.g., 0.1)
                num_layers,           # Number of encoder/decoder layers (e.g., 5)
                max_sequence_length,  # Max characters per sentence (e.g., 200)
                kn_vocab_size,        # Number of unique Kannada characters (output vocabulary)
                english_to_index,     # Dict: English character → integer index
                kannada_to_index,     # Dict: Kannada character → integer index
                START_TOKEN,          # Special start-of-sentence token
                END_TOKEN,            # Special end-of-sentence token
                PADDING_TOKEN         # Special padding token
                ):
        super().__init__()
        # Encoder: processes English input sentences
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        # Decoder: generates Kannada output conditioned on encoder output
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        # Final linear layer: maps d_model → kn_vocab_size to predict the next Kannada character
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x,                                    # Batch of English sentences (list of strings)
                y,                                    # Batch of Kannada sentences (list of strings)
                encoder_self_attention_mask=None,      # Mask: padding in English input
                decoder_self_attention_mask=None,      # Mask: padding + look-ahead in Kannada input
                decoder_cross_attention_mask=None,     # Mask: padding when decoder attends to encoder
                enc_start_token=False,                 # English typically doesn't need START token
                enc_end_token=False,                   # English typically doesn't need END token
                dec_start_token=False,                 # Should be True — decoder input starts with START
                dec_end_token=False):                  # Decoder input typically doesn't include END token
        """
        Full forward pass of the Transformer.

        Args:
            x: Batch of English sentences
            y: Batch of Kannada sentences (teacher-forced during training)
            encoder_self_attention_mask: Padding mask for encoder
            decoder_self_attention_mask: Padding + causal mask for decoder
            decoder_cross_attention_mask: Padding mask for cross-attention
            enc_start_token / enc_end_token: Whether to add special tokens to English
            dec_start_token / dec_end_token: Whether to add special tokens to Kannada

        Returns:
            Logits tensor — (batch, seq_len, kn_vocab_size)
            Each position contains scores for every Kannada character.
            Apply softmax to get probabilities, or use directly with CrossEntropyLoss.
        """
        # Step 1: Encode English input → context-aware embeddings
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        # Step 2: Decode Kannada output conditioned on English context
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        # Step 3: Project decoder output to vocabulary size for character prediction
        out = self.linear(out)
        return out