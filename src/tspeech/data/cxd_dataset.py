from os import path
import ast

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class CXDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: str, sr: int = 16000):
        self.df = df
        self.dataset_dir = dataset_dir
        self.sr = sr

        # Resample will be determined dynamically based on audio file sample rate
        self.resample_cache = {}

    def _get_resampler(self, orig_freq: int):
        """Get or create resampler for a specific sample rate"""
        if orig_freq not in self.resample_cache:
            self.resample_cache[orig_freq] = torchaudio.transforms.Resample(
                orig_freq=orig_freq, new_freq=self.sr
            )
        return self.resample_cache[orig_freq]

    def __len__(self) -> int:
        return len(self.df)

    def _parse_trust_label(self, trust_label):
        """Parse trust_label which can be a list string like '[1, 1, 0]' or a single value"""
        if pd.isna(trust_label):
            return 0.0
        
        # If it's a string that looks like a list, parse it
        if isinstance(trust_label, str) and trust_label.startswith('['):
            try:
                parsed = ast.literal_eval(trust_label)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Use the first value or majority vote
                    return float(parsed[0])
                return float(parsed) if isinstance(parsed, (int, float)) else 0.0
            except:
                return 0.0
        
        # Otherwise, convert to float directly
        try:
            return float(trust_label)
        except:
            return 0.0

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        data = self.df.iloc[i].to_dict()
        
        # Get the name from CSV (e.g., "p322p323-part2_ch2:437.709410:438.630450:q14:n1:F.txt")
        name = data['name']
        
        # Convert name to wav filename - replace .txt with .wav
        # The name format is like: p322p323-part2_ch2:437.709410:438.630450:q14:n1:F.txt
        # We need to find the matching wav file
        wav_filename = name.replace('.txt', '.wav')
        
        # Try direct match first
        wav_path = path.join(self.dataset_dir, "wav", wav_filename)
        
        # If not found, try to find by base part (before first colon)
        if not path.exists(wav_path):
            base_part = name.split(':')[0]  # e.g., "p322p323-part2_ch2"
            import os
            wav_dir = path.join(self.dataset_dir, "wav")
            if os.path.exists(wav_dir):
                # Look for files containing the base part
                for f in os.listdir(wav_dir):
                    if f.endswith('.wav') and base_part in f:
                        wav_path = path.join(wav_dir, f)
                        break
        
        # Load audio
        wav, orig_sr = torchaudio.load(wav_path)
        
        # Resample if needed
        if orig_sr != self.sr:
            resampler = self._get_resampler(orig_sr)
            wav = resampler(wav)

        mask = torch.ones_like(wav, dtype=torch.bool)
        
        # Parse trust_label
        trust_value = self._parse_trust_label(data['trust_label'])
        trustworthy = torch.tensor([[trust_value]], dtype=torch.float)

        return wav, mask, trustworthy

