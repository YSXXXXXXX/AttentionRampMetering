import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_head: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
        query, key, value = self._split(query), self._split(key), self._split(value)
        attn_score, attn = self.attention(query, key, value)
        attn = self._concat(attn)
        out = self.w_o(attn)
        return out

    def _concat(self, tensor: torch.Tensor):
        # input shape: (batch size, num_head, length, d_block)
        tensor = tensor.transpose(1, 2)  # (batch size, length, num_head, d_block)
        tensor = tensor.reshape(tensor.size()[0], tensor.size()[1], self.model_dim)  # (batch size, length, d_model)
        return tensor

    def _split(self, tensor: torch.Tensor):
        batch_size, length, d_model = tensor.size()
        d_block = d_model // self.num_head
        # (batch size, length, d_model) -> (batch size, length, num_head, d_block)
        tensor = tensor.view(batch_size, length, self.num_head, d_block)
        # (batch size, num_head, length, d_block)
        tensor.transpose_(1, 2)
        return tensor

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # scale dot product attention
        d_block = query.size()[-1]
        key_t = key.transpose(2, 3)
        # attention score = softmax(q * k^T / sqrt(d_block)). shape: (batch size, num_head, length, length)
        attn_score = torch.softmax(torch.matmul(query, key_t) / math.sqrt(d_block), dim=-1)
        attn = torch.matmul(attn_score, value)  # (batch size, num_head, length, d_block)
        return attn_score, attn


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, num_head: int, feed_forward_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_head)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, model_dim)
        )
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        residual = query
        out = self.attention(query, key, value)
        out = self.dropout1(out)
        out = self.layer_norm1(out + residual)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.layer_norm2(out + residual)
        return out


class DownstreamTask(nn.Module):
    def __init__(self, num_mainline: int, num_ramp: int, feature_dim: int, output_dim: int, hidden_dim: int,
                 embedding_dim: int, feed_forward_dim: int, num_attend_layer: int, num_head: int, dropout: float):
        super().__init__()
        self.num_mainline, self.num_ramp = num_mainline, num_ramp
        self.feature_dim = feature_dim
        self.model_dim = embedding_dim
        self.input_proj = nn.Linear(feature_dim, embedding_dim)
        self.mainline_attention = nn.ModuleList(
            [AttentionLayer(self.model_dim, num_head, feed_forward_dim, dropout) for _ in range(num_attend_layer)]
        )
        self.ramp_attention = nn.ModuleList(
            [AttentionLayer(self.model_dim, num_head, feed_forward_dim, dropout) for _ in range(num_attend_layer)]
        )
        self.mix_attn_dim = (num_mainline + num_ramp) * self.model_dim
        self.output_proj = nn.Sequential(
            nn.Linear(self.mix_attn_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state: torch.Tensor):
        # input state: (batch size, num_mainline + num_ramp, feature_dim)
        # mainline state: [:num_mainline * feature_dim], ramp state: [-num_ramp * feature_dim:]
        mainline_state, ramp_state = state[:, :self.num_mainline, :], state[:, -self.num_ramp:, :]
        mainline_state = self.input_proj(mainline_state)  # (batch size, num_mainline, embedding_dim)
        ramp_state = self.input_proj(ramp_state)  # (batch size, num_ramp, embedding_dim)
        mainline_attn, ramp_attn = mainline_state, ramp_state
        for mainline_attn_layer, ramp_attn_layer in zip(self.mainline_attention, self.ramp_attention):
            tmp_mainline_attn = mainline_attn_layer(mainline_attn, ramp_attn, ramp_attn)
            tmp_ramp_attn = ramp_attn_layer(ramp_attn, mainline_attn, mainline_attn)
            mainline_attn, ramp_attn = tmp_mainline_attn, tmp_ramp_attn
        # (batch size, num_mainline + num_ramp, model_dim)
        mix_attn = torch.concatenate((mainline_attn, ramp_attn), dim=1).view(-1, self.mix_attn_dim)
        output = self.output_proj(mix_attn)
        return output


class AttentionPolicy(DownstreamTask):
    def __init__(self, num_mainline: int, num_ramp: int, feature_dim: int, action_dim: int, hidden_dim: int,
                 embedding_dim: int, feed_forward_dim: int, num_attend_layer: int, num_head: int, dropout: float):
        super().__init__(num_mainline, num_ramp, feature_dim, action_dim, hidden_dim,
                         embedding_dim, feed_forward_dim, num_attend_layer, num_head, dropout)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_distribution(self, state: torch.Tensor):
        state = state.view(state.size()[0], -1, self.feature_dim)
        mean = self.forward(state)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)


class AttentionCritic(DownstreamTask):
    def __init__(self, num_mainline: int, num_ramp: int, feature_dim: int, hidden_dim: int,
                 embedding_dim: int, feed_forward_dim: int, num_attend_layer: int, num_head: int, dropout: float):
        super().__init__(num_mainline, num_ramp, feature_dim, 1, hidden_dim, embedding_dim, feed_forward_dim,
                         num_attend_layer, num_head, dropout)

    def get_value(self, state: torch.Tensor):
        state = state.view(state.size()[0], -1, self.feature_dim)
        value = self.forward(state)
        return value
