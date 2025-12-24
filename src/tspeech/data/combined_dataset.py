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
        
        # Verify all datasets in CSV have corresponding base directories (or use full paths)
        datasets_in_csv = self.df['dataset'].unique()
        # Check if filenames are full paths
        if len(self.df) > 0:
            sample_filename = self.df['filename'].iloc[0]
            if path.isabs(sample_filename):
                # Full paths in CSV - base_dirs not strictly needed
                pass
            else:
                # Relative paths - need base_dirs
                missing_dirs = [ds for ds in datasets_in_csv if ds not in base_dirs or not base_dirs[ds]]
                if missing_dirs:
                    raise ValueError(
                        f"CSV contains relative paths for datasets without base directories: {missing_dirs}. "
                        f"Either provide base directories or use full paths in CSV."
                    )

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
        base_dir = self.base_dirs.get(dataset, "")
        
        # Construct full path - if filename is already absolute or base_dir is empty, use filename as-is
        if path.isabs(filename) or not base_dir:
            full_path = filename
        else:
            full_path = path.join(base_dir, filename)
        
        # Verify file exists with helpful error message
        if not path.exists(full_path):
            raise FileNotFoundError(
                f"Audio file not found!\n"
                f"  Row {i}: {filename}\n"
                f"  Dataset: {dataset}\n"
                f"  Base dir: {base_dir}\n"
                f"  Full path: {full_path}\n"
                f"  Base dirs available: {list(self.base_dirs.keys())}\n"
                f"Please check:\n"
                f"  1. File exists at: {full_path}\n"
                f"  2. Base directory is correct: {base_dir}\n"
                f"  3. CSV filename column has correct paths"
            )
        
        # Load audio with error handling
        # Print file path for debugging if loading fails
        try:
            wav, orig_sr = torchaudio.load(full_path)
        except Exception as e:
            torchaudio_error = str(e) if str(e) else f"{type(e).__name__}"
            # Try alternative loading methods
            try:
                import soundfile as sf
                import numpy as np
                # Print file path before attempting soundfile load (in case it fails)
                print(f"\n[DEBUG] Attempting to load with soundfile: {full_path}", flush=True)
                wav_np, orig_sr = sf.read(full_path, dtype='float32')
                wav = torch.from_numpy(wav_np).float()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)  # Add channel dimension
                elif wav.dim() == 2 and wav.shape[0] > 1:
                    # If stereo, take first channel
                    wav = wav[0:1, :]
            except Exception as e2:
                # CRITICAL: Print file path IMMEDIATELY before any other operations
                # This ensures we see which file failed even if error object is corrupted
                import sys
                print("\n" + "=" * 60, file=sys.stderr, flush=True)
                print("ERROR: Failed to load audio file!", file=sys.stderr, flush=True)
                print("=" * 60, file=sys.stderr, flush=True)
                print(f"FAILED FILE PATH: {full_path}", file=sys.stderr, flush=True)
                print(f"CSV filename: {filename}", file=sys.stderr, flush=True)
                print(f"Row {i} in CSV (0-indexed, 1-indexed: {i+1})", file=sys.stderr, flush=True)
                print(f"Dataset: {dataset}", file=sys.stderr, flush=True)
                print(f"Base directory: {base_dir}", file=sys.stderr, flush=True)
                
                # Handle soundfile.LibsndfileError specifically
                import soundfile
                soundfile_error_type = type(e2).__name__
                
                # Get more info about the file
                import os
                file_size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                file_info = f"File size: {file_size} bytes"
                try:
                    import subprocess
                    file_type = subprocess.check_output(['file', full_path], stderr=subprocess.STDOUT).decode().strip()
                    file_info += f"\n  File type: {file_type}"
                    print(f"File info: {file_info}", file=sys.stderr, flush=True)
                except Exception as fe:
                    print(f"Could not get file info: {fe}", file=sys.stderr, flush=True)
                
                # Try to get error message (LibsndfileError sometimes fails on str())
                try:
                    error_msg = str(e2) if str(e2) else f"{soundfile_error_type} (unable to get error details)"
                    print(f"Error: {error_msg}", file=sys.stderr, flush=True)
                except Exception:
                    error_msg = f"{soundfile_error_type} (exception stringification failed - file likely corrupted)"
                    print(f"Error: {error_msg}", file=sys.stderr, flush=True)
                print("=" * 60, file=sys.stderr, flush=True)
                
                if isinstance(e2, soundfile.LibsndfileError):
                    # Try librosa as last resort
                    try:
                        import librosa
                        wav_np, orig_sr = librosa.load(full_path, sr=None, mono=True)
                        wav = torch.from_numpy(wav_np).float()
                        wav = wav.unsqueeze(0)  # Add channel dimension
                    except Exception as e3:
                        librosa_error = str(e3) if str(e3) else f"{type(e3).__name__}"
                        error_details = (
                            f"Failed to load audio file (LibsndfileError)!\n"
                            f"  File: {full_path}\n"
                            f"  CSV filename: {filename}\n"
                            f"  Row {i} in CSV (0-indexed, 1-indexed: {i+1})\n"
                            f"  Dataset: {dataset}\n"
                            f"  Base directory: {base_dir}\n"
                            f"  {file_info}\n"
                            f"  torchaudio error: {torchaudio_error}\n"
                            f"  soundfile error: {error_msg}\n"
                            f"  librosa error: {librosa_error}\n"
                            f"\nTroubleshooting:\n"
                            f"  1. Check file exists: ls -lh '{full_path}'\n"
                            f"  2. Check file format: file '{full_path}'\n"
                            f"  3. File may be corrupted - try: ffmpeg -i '{full_path}' test_output.wav\n"
                            f"  4. Remove this file from CSV or fix the file\n"
                            f"  5. Run validation script: python fix_audio_csv.py --combined_csv combined_dataset.csv\n"
                            f"  6. Install/update libraries: pip install --upgrade soundfile librosa"
                        )
                        # Print error to stderr for immediate visibility
                        import sys
                        print("\n" + "=" * 60, file=sys.stderr)
                        print("ERROR: Failed to load audio file!", file=sys.stderr)
                        print("=" * 60, file=sys.stderr)
                        print(error_details, file=sys.stderr)
                        print("=" * 60 + "\n", file=sys.stderr)
                        raise RuntimeError(error_details) from e3
                else:
                    # Try librosa as last resort for other errors
                    try:
                        import librosa
                        wav_np, orig_sr = librosa.load(full_path, sr=None, mono=True)
                        wav = torch.from_numpy(wav_np).float()
                        wav = wav.unsqueeze(0)  # Add channel dimension
                    except Exception as e3:
                        librosa_error = str(e3) if str(e3) else f"{type(e3).__name__}"
                        raise RuntimeError(
                            f"Failed to load audio file!\n"
                            f"  File: {full_path}\n"
                            f"  CSV filename: {filename}\n"
                            f"  Row {i} in CSV (0-indexed, 1-indexed: {i+1})\n"
                            f"  Dataset: {dataset}\n"
                            f"  Base directory: {base_dir}\n"
                            f"  {file_info}\n"
                            f"  torchaudio error: {torchaudio_error}\n"
                            f"  soundfile error: {error_msg}\n"
                            f"  librosa error: {librosa_error}\n"
                            f"\nTroubleshooting:\n"
                            f"  1. Check file exists: ls -lh '{full_path}'\n"
                            f"  2. Check file format: file '{full_path}'\n"
                            f"  3. Run validation script: python fix_audio_csv.py --combined_csv combined_dataset.csv\n"
                            f"  4. Install libraries: pip install soundfile librosa\n"
                            f"  5. Remove corrupted file from CSV or fix the file"
                        ) from e3
        
        # Resample if needed
        if orig_sr != self.sr:
            resampler = self._get_resampler(orig_sr)
            wav = resampler(wav)
        
        # Create mask (all ones for now, assuming no masking needed)
        mask = torch.ones_like(wav, dtype=torch.bool)
        
        # Create label tensor
        trustworthy = torch.tensor([[label]], dtype=torch.float)
        
        return wav, mask, trustworthy


