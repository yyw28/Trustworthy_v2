import logging
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from tspeech.model.tacotron2 import Attention


class Decoder(nn.Module):
    def __init__(
        self,
        num_mels: int,
        embedding_dim: int,
        prenet_dim: int,
        att_rnn_dim: int,
        att_dim: int,
        rnn_hidden_dim: int,
        dropout: float,
        extra_att_in_dim: int = 0,
        extra_decoder_in_dim: int = 0,
    ):
        super().__init__()

        # Attention components - a LSTM cell and attention module
        self.att_rnn = nn.LSTMCell(
            prenet_dim + embedding_dim + extra_att_in_dim, att_rnn_dim
        )

        self.att_rnn_dropout = nn.Dropout(0.1)

        self.attention = Attention(
            attention_rnn_dim=att_rnn_dim,
            embedding_dim=embedding_dim,
            attention_dim=att_dim,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
        )

        # Decoder LSTM cell
        self.lstm = nn.LSTMCell(
            att_rnn_dim + embedding_dim + extra_decoder_in_dim, rnn_hidden_dim
        )
        self.lstm_dropout = nn.Dropout(0.1)

        # Final layer producing Mel output
        self.mel_out = nn.Linear(
            rnn_hidden_dim + embedding_dim + extra_decoder_in_dim, num_mels
        )

        # Final layer producing gate output
        self.gate = nn.Linear(rnn_hidden_dim + embedding_dim, 1)

    def forward(
        self,
        prev_mel_prenet: Tensor,
        att_rnn_hidden: Tuple[Tensor, Tensor],
        att_context,
        att_weights,
        att_weights_cum,
        rnn_hidden: Tuple[Tensor, Tensor],
        encoded,
        att_encoded,
        encoded_mask,
        speech_features: Optional[Tensor] = None,
        extra_att_in: Optional[Tensor] = None,
        extra_decoder_in: Optional[Tensor] = None,
    ):
        # Attention -------------------------------------------------------------------------------
        # Attention RNN
        att_rnn_input = [prev_mel_prenet, att_context]
        if extra_att_in is not None:
            att_rnn_input.append(extra_att_in)

        att_h, att_c = self.att_rnn(torch.concat(att_rnn_input, -1), att_rnn_hidden)
        att_h = self.att_rnn_dropout(att_h)

        # Attention module
        att_weights_cat = torch.concat(
            [att_weights.unsqueeze(1), att_weights_cum.unsqueeze(1)], 1
        )

        att_context, att_weights = self.attention(
            attention_hidden_state=att_h,
            memory=encoded,
            processed_memory=att_encoded,
            attention_weights_cat=att_weights_cat,
            mask=encoded_mask,
        )

        # Save cumulative attention weights
        att_weights_cum += att_weights

        # Decoder ---------------------------------------------------------------------------------
        # Run attention output through the decoder RNN
        decoder_input = [att_h, att_context]
        if speech_features is not None:
            decoder_input.append(speech_features)
        if extra_decoder_in is not None:
            decoder_input.append(extra_decoder_in)

        decoder_in = torch.concat(decoder_input, -1)

        rnn_h, rnn_c = self.lstm(decoder_in, rnn_hidden)
        rnn_h = self.lstm_dropout(rnn_h)

        rnn_out = [rnn_h, att_context]
        gate_out = self.gate(torch.cat(rnn_out, -1))

        if extra_decoder_in is not None:
            rnn_out.append(extra_decoder_in)

        mel_out = self.mel_out(torch.cat(rnn_out, -1))

        return (
            mel_out,
            gate_out,
            (att_h, att_c),
            att_context,
            att_weights,
            att_weights_cum,
            (rnn_h, rnn_c),
        )
