"""Wrapper for Vocos vocoder to convert mel spectrograms to waveforms."""

import torch
from torch import Tensor
from typing import Optional


class VocosVocoder:
    """
    Wrapper for Vocos vocoder.
    
    Converts mel spectrograms to waveforms for RL training.
    """
    
    def __init__(
        self,
        model_name: str = "charactr/vocos-mel-24khz",
        sample_rate: int = 22050,
        device: Optional[str] = None,
    ):
        """
        Initialize Vocos vocoder.
        
        Parameters
        ----------
        model_name : str
            HuggingFace model name for Vocos
            Options:
            - "charactr/vocos-mel-24khz" (24kHz, for 24kHz audio)
            - "charactr/vocos-mel-16khz" (16kHz, for 16kHz audio)
        sample_rate : int
            Target sample rate (should match model)
        device : Optional[str]
            Device to run on (None = auto-detect)
        """
        try:
            from vocos import Vocos
        except ImportError:
            raise ImportError(
                "vocos not installed. Install with: pip install vocos"
            )
        
        self.sample_rate = sample_rate
        
        # Auto-detect device (use CPU for vocoder due to MPS limitations)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Note: MPS has limitations with some operations, so we use CPU for vocoder
        
        self.device = torch.device(device)
        
        # Load Vocos model
        print(f"Loading Vocos vocoder: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Sample rate: {sample_rate} Hz")
        
        try:
            self.vocoder = Vocos.from_pretrained(model_name)
            self.vocoder = self.vocoder.to(self.device)
            self.vocoder.eval()
            print(f"  ✓ Vocoder loaded successfully")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load pretrained model: {e}")
            print(f"  Using fallback Griffin-Lim vocoder")
            self.vocoder = None
    
    def __call__(
        self,
        mel_spectrogram: Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tensor:
        """
        Convert mel spectrogram to waveform.
        
        Parameters
        ----------
        mel_spectrogram : Tensor
            Mel spectrogram of shape (batch, time_frames, num_mels)
            or (batch, num_mels, time_frames)
        sample_rate : Optional[int]
            Sample rate (uses self.sample_rate if None)
            
        Returns
        -------
        Tensor
            Waveform of shape (batch, samples)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure mel is on correct device
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        # Handle different input shapes - ensure (batch, time, mels)
        if mel_spectrogram.dim() == 3:
            if mel_spectrogram.shape[2] == 80:
                # Already (batch, time, mels) - good
                pass
            elif mel_spectrogram.shape[1] == 80:
                # (batch, mels, time) -> (batch, time, mels)
                mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # Use Griffin-Lim directly (more compatible with 80 mel channels)
        # Vocos models typically expect 100 channels, so we'll use Griffin-Lim
        return self._griffin_lim_from_mel(mel_spectrogram, sample_rate)
    
    def _griffin_lim_from_mel(
        self,
        mel_spectrogram: Tensor,
        sample_rate: int,
    ) -> Tensor:
        """
        Convert mel spectrogram to waveform using Griffin-Lim.
        
        This converts mel -> linear spectrogram -> waveform.
        
        Parameters
        ----------
        mel_spectrogram : Tensor
            Mel spectrogram (batch, time, mels) with 80 mel channels
        sample_rate : int
            Sample rate
            
        Returns
        -------
        Tensor
            Waveform (batch, samples)
        """
        import torchaudio
        
        # Ensure shape is (batch, time, mels)
        if mel_spectrogram.dim() == 3:
            if mel_spectrogram.shape[2] != 80:
                # Try to fix shape
                if mel_spectrogram.shape[1] == 80:
                    mel_spectrogram = mel_spectrogram.transpose(1, 2)
        
        # Move to CPU for vocoder operations (MPS has limitations)
        cpu_device = torch.device("cpu")
        mel_spectrogram = mel_spectrogram.to(cpu_device)
        
        # Convert mel to linear spectrogram
        # Parameters matching Tacotron2 defaults
        n_fft = 2048
        hop_length = 256
        win_length = 1024
        n_mels = 80
        
        # Create mel-to-linear converter (on CPU)
        mel_to_linear = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=sample_rate // 2,
        ).to(cpu_device)
        
        # Convert mel -> linear spectrogram
        # Input: (batch, time, mels) -> (batch, mels, time) for InverseMelScale
        mel_for_conversion = mel_spectrogram.transpose(1, 2)  # (batch, mels, time)
        
        # Convert to linear spectrogram magnitude
        linear_spec = mel_to_linear(mel_for_conversion)  # (batch, freq, time)
        
        # Griffin-Lim to convert magnitude spectrogram -> waveform
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=32,
        ).to(cpu_device)
        
        # Convert to waveform
        waveform = griffin_lim(linear_spec)  # (batch, samples)
        
        return waveform  # Already on CPU

