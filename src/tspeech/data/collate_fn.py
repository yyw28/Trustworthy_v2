import torch
from torch import Tensor, nn


def trustworthy_collate_fn(
    data: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    wavs: list[Tensor] = []
    masks: list[Tensor] = []
    trustworthies: list[Tensor] = []

    for wav, mask, trustworthy in data:
        wavs.append(wav.squeeze(0))
        masks.append(mask.squeeze(0))
        trustworthies.append(trustworthy.squeeze(0))

    return (
        nn.utils.rnn.pad_sequence(wavs, batch_first=True),
        nn.utils.rnn.pad_sequence(masks, batch_first=True),
        torch.stack(trustworthies),
    )
