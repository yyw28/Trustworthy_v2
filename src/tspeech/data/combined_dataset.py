from os import path

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    """Dataset that loads audio files from a combined CSV with filename and label columns."""
    
    def __init__(self, csv_path: str, base_dirs: dict[str, str], sr: int = 16000):
        """
        Parameters
        ----------
        csv_path : str
            Path to the combined CSV file with columns: filename, label, dataset
        base_dirs : dict[str, str]
            Dictionary mapping dataset names to their base directories
            e.g., {"tis": "/path/to/tis", "cxd": "/path/to/cxd"}
        sr : int
            Target sample rate (default: 16000)
        """
        self.df = pd.read_csv(csv_path)
        self.base_dirs = base_dirs
        self.sr = sr
        
        # Cache for resamplers (different audio files may have different sample rates)
        self.resample_cache = {}
        
        # Verify required columns exist
        required_cols = ['filename', 'label', 'dataset']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV file must contain columns: {required_cols}. Missing: {missing_cols}")
        
        # Filter out rows with invalid labels
        self.df = self.df[self.df['label'] != -1]
        self.df = self.df[self.df['label'].notna()]
        
        # Verify all datasets in CSV have corresponding base directories
        datasets_in_csv = self.df['dataset'].unique()
        missing_dirs = [ds for ds in datasets_in_csv if ds not in base_dirs]
        if missing_dirs:
            raise ValueError(f"CSV contains datasets without base directories: {missing_dirs}")

    def _get_resampler(self, orig_freq: int):
        """Get or create resampler for a specific sample rate"""
        if orig_freq not in self.resample_cache:
            self.resample_cache[orig_freq] = torchaudio.transforms.Resample(
                orig_freq=orig_freq, new_freq=self.sr
            )
        return self.resample_cache[orig_freq]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        row = self.df.iloc[i]
        
        filename = row['filename']
        dataset = row['dataset']
        label = int(row['label'])
        
        # Get the base directory for this dataset
        base_dir = self.base_dirs[dataset]
        
        # Construct full path
        full_path = path.join(base_dir, filename)
        
        # Load audio
        wav, orig_sr = torchaudio.load(full_path)
        
        # Resample if needed
        if orig_sr != self.sr:
            resampler = self._get_resampler(orig_sr)
            wav = resampler(wav)
        
        # Create mask (all ones for now, assuming no masking needed)
        mask = torch.ones_like(wav, dtype=torch.bool)
        
        # Create label tensor
        trustworthy = torch.tensor([[label]], dtype=torch.float)
        
        return wav, mask, trustworthy


