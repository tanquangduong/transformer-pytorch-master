import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    """
    An input embedding layer that maps input tokens to embeddings.

    Parameters:
    vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens.
    d_model (int): The dimensionality of the embeddings.
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, d_model)  
        self.d_model = d_model  

    def forward(self, x):  # x: [batch_size, seq_len]
        # Apply the embedding layer to the input,
        # and then scales the result by the square root of the embedding dimensionality.
        return self.embedding(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model].

class PositionalEncoding(nn.Module):
    """
    A positional encoding layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the embeddings.
    seq_len (int): The maximum sequence length.
    dropout (float): The dropout rate.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()  
        self.dropout = nn.Dropout(dropout)  

        # Compute the positional encodings once in log space.
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        base = 10000.0 ** (-1.0 / d_model)
        div_term = torch.pow(base, torch.arange(0, d_model, 2).float())

        # Compute the positional encodings 
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register the positional encodings as a buffer.
        self.register_buffer("pe", pe)

    def forward(self, x):  # x(embeded sequence): [batch_size, seq_len, d_model]
        # Add the positional encodings to the input embeddings,
        # Then apply dropout to the result.
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)

        # self.pe[:, :x.shape[1], :] is to adapt the shape of decoder input in case of traning or inference
        return self.dropout(x) # [batch_size, seq_len, d_model].

class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the embeddings.
    h (int): The number of attention heads.
    dropout (float): The dropout rate.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()  # Call the constructor of the parent class.
        self.d_model = d_model
        self.h = h

        # Ensure that the embedding dimensionality is divisible by the number of heads.
        assert d_model % h == 0, "d_model % num_heads should be zero"
        self.d_k = d_model // h

        # Initialize the linear transformations for the queries, keys, values, and output.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Initialize the dropout layer.
        self.dropout = nn.Dropout(dropout)

    # The attention function calculates the attention scores for a given query, key, and value.
    @staticmethod
    def attention(query_k, key_k, value_k, d_k, mask=None, dropout=nn.Dropout):
        # query_k | key_k | value_k: [batch_size, h, seq_len, d_k]
        # mask: [batch_size, 1, seq_len, seq_len]

        # Calculate the attention scores.
        attention_score = (query_k @ key_k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply the mask, if provided.
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        # Apply the softmax and dropout function to the attention scores.
        attention_score = torch.softmax(attention_score, dim=-1)
        attention_score = dropout(attention_score)

        # Return the weighted sum of the values, along with the attention scores.
        # [batch_size, h, seq_len, d_k], [batch_size, h, seq_len, seq_len]
        return (attention_score @ value_k, attention_score)

    # The forward method is called when we pass input data into this layer.
    def forward(self, query, key, value, mask=None):
        # query | key | value: [batch_size, seq_len, d_model]
        # mask: [batch_size, 1, seq_len, seq_len]

        # Apply the linear transformations to the query, key, and value.
        query_k = self.w_q(query)
        key_k = self.w_k(key)
        value_k = self.w_v(value)

        # Reshape and transpose the queries, keys, and values to prepare them for multi-head attention.
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, h, d_k] -> [batch_size, h, seq_len, d_k]
        query_k = query_k.view(query_k.shape[0], query_k.shape[1], self.h, self.d_k).transpose(1, 2)
        key_k = key_k.view(key_k.shape[0], key_k.shape[1], self.h, self.d_k).transpose(1, 2)
        value_k = value_k.view(value_k.shape[0], value_k.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate the attention.
        attention, _ = self.attention(query_k, key_k, value_k, self.d_k, mask, self.dropout)

        # Concatenate the attention heads.
        # [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k] -> [batch_size, seq_len, d_model]
        attention = (attention.transpose(1, 2)
            .contiguous()
            .view(attention.shape[0], -1, self.d_model)
        )

        # Apply the output linear transformation and return the result.
        return self.w_o(attention)  # [batch_size, seq_len, d_model]
    

class FeedForward(nn.Module):
    """
    A feed-forward network layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    d_ff (int): The dimensionality of the hidden layer.
    dropout (float): The dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()  
        
        # Initialize the first linear transformation, which increases the dimensionality to d_ff.
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Initialize the second linear transformation, which decreases the dimensionality back to d_model.
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Initialize the dropout layer.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [batch_size, seq_len, d_model]
        # It applies the first linear transformation, applies the ReLU activation function,
        # applies dropout, and then applies the second linear transformation.
        return self.linear2(
            self.dropout(torch.relu(self.linear1(x))) # [batch_size, seq_len, d_model].
        )


class LayerNorm(nn.Module):
    """
    A layer normalization layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    eps (float): A small number to prevent division by zero. Default is 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()  
        
        # Initialize the scale and shift parameters, which are learnable.
        self.para_mul = nn.Parameter(torch.ones(d_model))
        self.para_bias = nn.Parameter(torch.zeros(d_model))
        
        # Store the epsilon value.
        self.eps = eps

    def forward(self, x):  # x: [batch_size, seq_len, d_model]
        # It calculates the mean and standard deviation of the input,
        # and then normalizes the input by subtracting the mean and dividing by the standard deviation.
        # It then scales and shifts the result using the learnable parameters.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (
            self.para_mul * (x - mean) / (std + self.eps) + self.para_bias # [batch_size, seq_len, d_model].
        )


class ResidualConnection(nn.Module):
    """
    A residual connection layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input and output.
    dropout (float): The dropout rate.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        
        # Initialize the layer normalization layer.
        self.norm = LayerNorm(d_model)
        
        # Initialize the dropout layer.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  # x: [batch_size, seq_len, d_model]

        # It applies layer normalization to the input, passes the result through the sublayer,
        return x + self.dropout(
            sublayer(self.norm(x)) # [batch_size, seq_len, d_model].
        )


class Projection(nn.Module):
    """
    A projection layer for use in a transformer model.

    Parameters:
    d_model (int): The dimensionality of the input.
    vocab_size (int): The size of the vocabulary, which is also the dimensionality of the output.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()  
        
        # Initialize the linear transformation that projects the input into the vocabulary space.
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # x: [batch_size, seq_len, d_model]

        # It applies the linear transformation to the input.
        return self.projection(x) # [batch_size, seq_len, vocab_size].
