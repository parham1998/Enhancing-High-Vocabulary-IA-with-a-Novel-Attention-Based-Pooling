# =============================================================================
# Import required libraries
# =============================================================================
import copy
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import MultiheadAttention
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_decoder_layers,
                 feedforward_size,
                 dropout,
                 activation,
                 norm_first,
                 remove_self_attn,
                 keep_query_position):
        super().__init__()
        self.keep_query_position = keep_query_position

        decoder_layer = TransformerDecoderLayer(hidden_size,
                                                num_heads,
                                                feedforward_size,
                                                dropout,
                                                activation,
                                                norm_first,
                                                not remove_self_attn)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_decoder_layers,
                                          decoder_norm)
        self.reset_parameters()

        if remove_self_attn:
            self.remove_self_attn_func()

    def reset_parameters(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def remove_self_attn_func(self):
        for layer in self.decoder.layers:
            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

    def forward(self, features, queries):
        bs, c, h, w = features.shape
        # (batch-size, num-pixels, hidden-size)
        features = features.flatten(2).permute(0, 2, 1)
        # (batch-size, num-groups, hidden-size)
        queries = queries.unsqueeze(0).expand(bs, -1, -1)
        #
        if self.keep_query_position:
            hidden_state, attn_weights = self.decoder(torch.zeros_like(queries),
                                                      features,
                                                      queries)
        else:
            hidden_state, attn_weights = self.decoder(queries,
                                                      features,
                                                      None)
        return hidden_state, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                queries,
                encoder_features,
                query_position: Optional[Tensor] = None):
        output = queries

        for layer in self.layers:
            output, attn_weights = layer(output,
                                         encoder_features,
                                         query_position)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 feedforward_size,
                 dropout,
                 activation,
                 norm_first,
                 keep_self_attn):
        super().__init__()
        self.norm_first = norm_first
        self.keep_self_attn = keep_self_attn

        self.self_attn = MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.multihead_attn = MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_size, feedforward_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def with_position_encoding(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position

    # self-attention block
    def self_attention_block(self,
                             queries,
                             query_position: Optional[Tensor] = None):
        q = k = self.with_position_encoding(queries, query_position)
        x = self.self_attn(query=q,
                           key=k,
                           value=queries)[0]
        return self.dropout1(x)

    # cross-attention block
    def cross_attention_block(self,
                              queries,
                              features,
                              query_position: Optional[Tensor] = None):
        x, attn_weights = self.multihead_attn(query=self.with_position_encoding(queries,
                                                                                query_position),
                                              key=features,
                                              value=features)
        return self.dropout2(x), attn_weights

    # feedforward block
    def feedforward_block(self, queries):
        x = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        return self.dropout3(x)

    def forward(self,
                queries,
                features,
                query_position: Optional[Tensor] = None):
        x = queries
        if self.norm_first:
            if self.keep_self_attn:
                x = x + self.self_attention_block(self.norm1(x),
                                                  query_position)
            o, attn_weights = self.cross_attention_block(self.norm2(x),
                                                         features,
                                                         query_position)
            x = x + o
            x = x + self.feedforward_block(self.norm3(x))
        else:
            if self.keep_self_attn:
                x = self.norm1(
                    x + self.self_attention_block(x, query_position))
            o, attn_weights = self.cross_attention_block(x,
                                                         features,
                                                         query_position)
            x = self.norm2(x + o)
            x = self.norm3(x + self.feedforward_block(x))
        return x, attn_weights


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
