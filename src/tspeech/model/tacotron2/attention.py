import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        # Location layer
        padding = int((attention_location_kernel_size - 1) / 2)
        self.location_conv = nn.Conv1d(
            2,
            attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = nn.Linear(
            attention_location_n_filters, attention_dim, bias=False
        )

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
    ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        processed_query = self.query_layer(attention_hidden_state.unsqueeze(1))

        processed_attention_weights = self.location_conv(attention_weights_cat)
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_dense(processed_attention_weights)

        alignment = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )

        alignment = alignment.squeeze(-1)
        alignment.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights.detach()
