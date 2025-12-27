from typing import List, Optional
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from tspeech.data.tts import TTSBatch
from tspeech.model.tacotron2 import Tacotron2
from tspeech.model.tacotron2 import GST
from tspeech.model.tacotron2.bert_gst_encoder import BERTGSTEncoder
from tspeech.model.tacotron2.gst_with_weights import GSTWithWeights
from tspeech.model.tacotron2.rl_gst_policy import RLGSTPolicy, RLGSTRewardFunction


class TTSModel(pl.LightningModule):
    def __init__(
        self,
        num_chars: int,
        encoded_dim: int,
        encoder_kernel_size: int,
        num_mels: int,
        prenet_dim: int,
        att_rnn_dim: int,
        att_dim: int,
        rnn_hidden_dim: int,
        postnet_dim: int,
        dropout: float,
        speaker_tokens_enabled: bool = False,
        speaker_count: int = 1,
        max_len_override: Optional[int] = None,
        use_bert_gst: bool = True,
        bert_model_name: str = "bert-base-uncased",
        freeze_bert: bool = True,
        use_hubert_classifier: bool = True,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        hubert_checkpoint_path: Optional[str] = None,
        use_rl_training: bool = False,
        rl_temperature: float = 1.0,
        rl_entropy_coef: float = 0.01,
        use_vocoder: bool = True,
        vocoder_model_name: str = "charactr/vocos-mel-24khz",
        save_audio_dir: Optional[str] = None,
        save_audio_every_n_steps: int = 100,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.speaker_tokens = speaker_tokens_enabled
        self.max_len_override = max_len_override
        self.use_bert_gst = use_bert_gst
        self.use_hubert_classifier = use_hubert_classifier
        self.use_rl_training = use_rl_training
        self.rl_entropy_coef = rl_entropy_coef
        self.use_vocoder = use_vocoder
        self.save_audio_dir = save_audio_dir
        self.save_audio_every_n_steps = save_audio_every_n_steps
        
        # Create audio output directory if specified
        if self.save_audio_dir:
            Path(self.save_audio_dir).mkdir(parents=True, exist_ok=True)
        
        # Enable manual optimization when using multiple optimizers (RL training)
        if use_rl_training:
            self.automatic_optimization = False
        
        # Initialize vocoder for RL training
        self.vocoder = None
        if use_rl_training and use_vocoder:
            try:
                from tspeech.vocoder import VocosVocoder
                # Use CPU for vocoder (MPS has limitations with some operations)
                self.vocoder = VocosVocoder(
                    model_name=vocoder_model_name,
                    sample_rate=22050,  # Match TTS sample rate
                    device="cpu",  # Use CPU for compatibility
                )
                print("✓ Vocoder initialized for RL training (CPU mode)")
            except Exception as e:
                print(f"⚠ Warning: Could not initialize vocoder: {e}")
                print("  RL training will use placeholder loss")
                self.vocoder = None

        self.tacotron2 = Tacotron2(
            num_chars=num_chars,
            encoded_dim=encoded_dim,
            encoder_kernel_size=encoder_kernel_size,
            num_mels=num_mels,
            prenet_dim=prenet_dim,
            att_rnn_dim=att_rnn_dim,
            att_dim=att_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            postnet_dim=postnet_dim,
            dropout=dropout,
            speaker_tokens_enabled=speaker_tokens_enabled,
            speaker_count=speaker_count,
        )

        # BERT-based GST encoder (replaces mel-based GST)
        if use_bert_gst:
            self.bert_gst_encoder = BERTGSTEncoder(
                bert_model_name=bert_model_name,
                gst_token_num=10,
                freeze_bert=freeze_bert,
            )
            self.gst_with_weights = GSTWithWeights(
                token_num=10,
                token_embedding_size=encoded_dim,
            )
            
            # RL Policy network for GST weights optimization
            if use_rl_training:
                bert_hidden_size = self.bert_gst_encoder.bert.config.hidden_size
                self.rl_policy = RLGSTPolicy(
                    bert_hidden_size=bert_hidden_size,
                    gst_token_num=10,
                    temperature=rl_temperature,
                )
                self.reward_function = RLGSTRewardFunction(
                    trustworthiness_weight=1.0,
                    use_baseline=True,
                )
        else:
            # Fallback to original mel-based GST
            self.gst = GST(out_dim=encoded_dim)
        
        # HubERT trustworthiness classifier
        if use_hubert_classifier:
            from tspeech.hubert.model.htmodel import HTModel
            
            # Load from checkpoint
            self.hubert_classifier = HTModel.load_from_checkpoint(
                hubert_checkpoint_path,
                hubert_model_name=hubert_model_name,
                trainable_layers=0,  # Use frozen for inference
            )
            # Freeze for inference
            for param in self.hubert_classifier.parameters():
                param.requires_grad = False

    def forward(
        self,
        chars_idx: Tensor,
        chars_idx_len: Tensor,
        teacher_forcing: bool = True,
        mel_spectrogram: Optional[Tensor] = None,
        mel_spectrogram_len: Optional[Tensor] = None,
        speaker_id: Optional[Tensor] = None,
        max_len_override: Optional[int] = None,
        text: Optional[List[str]] = None,
        return_trustworthiness: bool = False,
    ):
        # Generate GST style embedding
        if self.use_bert_gst:
            # Use BERT to generate GST weights from text
            if text is None:
                raise ValueError("text is required when use_bert_gst=True")
            
            # Get GST weights (either from deterministic encoder or RL policy)
            if self.use_rl_training and self.training:
                # Use RL policy during training
                bert_embeddings = self.bert_gst_encoder.get_bert_embeddings(text)
                gst_weights, log_probs, entropy = self.rl_policy(
                    bert_embeddings,
                    deterministic=False,
                )
                # Store for REINFORCE loss computation
                self._rl_log_probs = log_probs
                self._rl_entropy = entropy
            else:
                # Use deterministic BERT encoder (inference or non-RL training)
                gst_weights = self.bert_gst_encoder(text)  # (batch, 10)
                self._rl_log_probs = None
                self._rl_entropy = None
            
            # Convert weights to style embedding
            style = self.gst_with_weights(gst_weights)  # (batch, encoded_dim)
            
            # Style will be added to encoded in Tacotron2, which expects (batch, seq_len, encoded_dim)
            # But Tacotron2 handles broadcasting, so we can pass (batch, encoded_dim)
            # and it will be added correctly. However, let's expand it to be explicit.
            batch_size = chars_idx.shape[0]
            seq_len = chars_idx.shape[1]
            style = style.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, encoded_dim)
        else:
            # Fallback to original mel-based GST
            # Extract style embedding from reference audio mel spectrogram
            # This style can be used with any text - the text determines WHAT is said,
            # while the GST style determines HOW it's said (prosody, emotion, etc.)
            if mel_spectrogram is None or mel_spectrogram_len is None:
                raise ValueError(
                    "mel_spectrogram and mel_spectrogram_len are required when use_bert_gst=False. "
                    "These should be the mel spectrogram from the reference audio for GST style extraction."
                )
            style = self.gst(mel_spectrogram, mel_spectrogram_len)  # (batch, encoded_dim)
            # Expand to match sequence length
            batch_size = chars_idx.shape[0]
            seq_len = chars_idx.shape[1]
            if style.dim() == 2:
                style = style.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, encoded_dim)

        # Forward through Tacotron2
        # Note: When use_bert_gst=False, mel_spectrogram is used for GST style extraction (already done above).
        # During inference (teacher_forcing=False), Tacotron2 doesn't use mel_spectrogram for teacher forcing,
        # but we still pass it here (it's ignored by Tacotron2 when teacher_forcing=False).
        # The text (chars_idx) determines WHAT is said, while the GST style (extracted from reference audio)
        # determines HOW it's said (prosody, emotion, speaking style, etc.).
        mel_output, mel_postnet, gate, alignment = self.tacotron2(
            chars_idx=chars_idx,
            chars_idx_len=chars_idx_len,
            teacher_forcing=teacher_forcing,
            mel_spectrogram=mel_spectrogram,  # Used for GST style extraction (when use_bert_gst=False)
            mel_spectrogram_len=mel_spectrogram_len,
            speaker_id=speaker_id,
            max_len_override=max_len_override,
            encoded_extra=style,  # GST style embedding (from reference audio when use_bert_gst=False)
        )
        
        # Evaluate trustworthiness with HubERT if requested
        trustworthiness_score = None
        if return_trustworthiness and self.use_hubert_classifier:
            try:
                # Convert mel_postnet to waveform using vocoder
                waveforms = self.vocoder(mel_postnet, sample_rate=22050)
                
                # Move waveforms to same device as model
                waveforms = waveforms.to(self.device)
                
                # Resample to 16 kHz for HubERT if needed
                if waveforms.shape[1] > 0:  # Check if waveform is not empty
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=22050,
                        new_freq=16000,
                    ).to(self.device)
                    waveforms_16k = resampler(waveforms)
                    
                    # Create attention mask
                    batch_size = waveforms_16k.shape[0]
                    seq_len = waveforms_16k.shape[1]
                    attention_mask = torch.ones(
                        batch_size, seq_len, dtype=torch.long, device=self.device
                    )
                    
                    # Get trustworthiness scores from HubERT
                    with torch.no_grad():
                        trustworthiness_logits = self.hubert_classifier(
                            wav=waveforms_16k,
                            mask=attention_mask,
                        )  # (batch_size, 1)
                        
                        # Convert logits to probability
                        trustworthiness_score = torch.sigmoid(trustworthiness_logits).squeeze(-1)  # (batch_size,)
            except Exception as e:
                # If vocoder or HubERT evaluation fails, return None
                print(f"⚠ Warning: Failed to compute trustworthiness score: {e}")
                trustworthiness_score = None
        
        if return_trustworthiness:
            return mel_output, mel_postnet, gate, alignment, trustworthiness_score
        else:
            return mel_output, mel_postnet, gate, alignment

    def validation_step(self, batch: TTSBatch, batch_idx):
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            text=batch.text,
        )

        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)

        loss = gate_loss + mel_loss + mel_post_loss
        self.log("val_mel_loss", loss, on_step=False, on_epoch=True)

        mel_spectrogram_len = batch.mel_spectrogram_len
        chars_idx_len = batch.chars_idx_len

        mel_spectrogram = batch.mel_spectrogram
        mel_spectrogram_pred = mel_spectrogram_post

        out = {
            "mel_spectrogram_pred": mel_spectrogram_pred[0, : mel_spectrogram_len[0]],
            "mel_spectrogram": mel_spectrogram[0, : mel_spectrogram_len[0]],
            "alignment": alignment[0, : mel_spectrogram_len[0], : chars_idx_len[0]],
            "gate": batch.gate[0],
            "gate_pred": gate[0],
        }

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx):
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=True,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            text=batch.text,
        )

        # Standard TTS losses
        gate_loss = F.binary_cross_entropy_with_logits(gate, batch.gate)
        mel_loss = F.mse_loss(mel_spectrogram, batch.mel_spectrogram)
        mel_post_loss = F.mse_loss(mel_spectrogram_post, batch.mel_spectrogram)
        
        tts_loss = gate_loss + mel_loss + mel_post_loss
        
        # RL loss computation (only during RL training)
        rl_loss = None
            # Full RL training with vocoder
        try:
            # Convert mel spectrogram to waveform
            waveforms = self.vocoder(mel_spectrogram_post, sample_rate=22050)
            
            # Save audio files periodically
            self._save_audio_if_needed(waveforms, batch_idx)
            
            # Compute RL loss from waveforms
            rl_loss = self.compute_rl_loss(waveforms, sample_rate=22050)
            
            # Log success
            if batch_idx % 10 == 0:
                self.log("rl_training_active", 1.0, on_step=True)
                self.log("vocoder_used", 1.0, on_step=True)
        except Exception as e:
            # Fallback on error
            print(f"⚠ Vocoder/RL error in batch {batch_idx}: {e}")
            rl_loss = torch.tensor(0.0, device=tts_loss.device, requires_grad=True)
        
        # Total loss
        loss = tts_loss + rl_loss

        # Manual optimization when using multiple optimizers (RL training)
        # PyTorch Lightning requires manual optimization when using multiple optimizers
        if self.automatic_optimization == False:
            optimizers = self.optimizers()
            if isinstance(optimizers, (list, tuple)) and len(optimizers) > 1:
                # Multiple optimizers: TTS and RL
                tts_opt, rl_opt = optimizers
                
                # TTS optimizer step
                tts_opt.zero_grad()
                tts_loss.backward(retain_graph=rl_loss is not None)
                tts_opt.step()
                
                # RL optimizer step (if RL loss exists)
                if rl_loss is not None:
                    rl_opt.zero_grad()
                    rl_loss.backward()
                    rl_opt.step()
            else:
                # Single optimizer (fallback)
                opt = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        # Log metrics (same for both manual and automatic optimization)
        self._log_training_metrics(gate_loss, mel_loss, mel_post_loss, rl_loss, loss)
        
        return loss if self.automatic_optimization else {"loss": loss.detach()}
    
    def _log_training_metrics(self, gate_loss, mel_loss, mel_post_loss, rl_loss, total_loss):
        """Log training metrics (shared between manual and automatic optimization)."""
        self.log("training_gate_loss", gate_loss.detach(), on_step=True, on_epoch=True)
        self.log("training_mel_loss", mel_loss.detach(), on_step=True, on_epoch=True)
        self.log("training_mel_post_loss", mel_post_loss.detach(), on_step=True, on_epoch=True)
        if rl_loss is not None:
            self.log("training_rl_loss", rl_loss.detach(), on_step=True, on_epoch=True)
        self.log("training_loss", total_loss.detach(), on_step=True, on_epoch=True)
    
    def _save_audio_if_needed(self, waveforms: Tensor, batch_idx: int):
        """Save audio files periodically if audio saving is enabled."""
        if not (self.save_audio_dir and batch_idx % self.save_audio_every_n_steps == 0):
            return
        
        try:
            import soundfile as sf
            import numpy as np
            
            for i, waveform in enumerate(waveforms):
                audio_np = waveform.detach().cpu().numpy()
                # Normalize to [-1, 1]
                max_val = np.abs(audio_np).max()
                if max_val > 0:
                    audio_np = audio_np / (max_val + 1e-8)
                
                # Save audio file
                audio_path = Path(self.save_audio_dir) / f"epoch_{self.current_epoch}_step_{batch_idx}_sample_{i}.wav"
                sf.write(str(audio_path), audio_np, 22050)
        except Exception as e:
            print(f"⚠ Warning: Failed to save audio: {e}")
    
    def compute_rl_loss(
        self,
        waveforms: Tensor,
        sample_rate: int = 16000,
    ) -> Tensor:
        # Move waveforms to same device as model
        waveforms = waveforms.to(self.device)
        """
        Compute REINFORCE loss for RL training.
        
        This should be called after generating waveforms from mel spectrograms
        using a vocoder.
        
        Parameters
        ----------
        waveforms : Tensor
            Generated waveforms of shape (batch_size, samples)
        sample_rate : int
            Sample rate of waveforms
            
        Returns
        -------
        Tensor
            REINFORCE loss
        """
        # Evaluate trustworthiness using HubERT
        batch_size = waveforms.shape[0]
        device = waveforms.device
        
        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            ).to(device)
            waveforms = resampler(waveforms)
        
        # Create attention mask
        seq_len = waveforms.shape[1]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Get trustworthiness scores
        with torch.no_grad():
            trustworthiness_logits = self.hubert_classifier(
                wav=waveforms,
                mask=attention_mask,
            )  # (batch_size, 1)
        
        # Compute rewards and advantages
        rewards, advantages = self.reward_function.compute_reward(trustworthiness_logits)
        
        # REINFORCE loss: -log_prob * advantage
        # We want to maximize reward, so minimize -log_prob * advantage
        reinforce_loss = -(self._rl_log_probs * advantages.detach()).mean()
        
        # Entropy bonus for exploration
        entropy_bonus = 0.0
        if self._rl_entropy is not None:
            entropy_bonus = -self.rl_entropy_coef * self._rl_entropy.mean()
        
        total_rl_loss = reinforce_loss + entropy_bonus
        
        # Log metrics
        self.log(
            "rl_reward_mean",
            rewards.mean().detach(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "rl_advantage_mean",
            advantages.mean().detach(),
            on_step=True,
            on_epoch=True,
        )
        if self._rl_entropy is not None:
            self.log(
                "rl_entropy_mean",
                self._rl_entropy.mean().detach(),
                on_step=True,
                on_epoch=True,
            )
        
        return total_rl_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        text = batch.text

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=batch.chars_idx,
            chars_idx_len=batch.chars_idx_len,
            teacher_forcing=False,
            speaker_id=batch.speaker_id,
            mel_spectrogram=batch.mel_spectrogram,
            mel_spectrogram_len=batch.mel_spectrogram_len,
            max_len_override=self.max_len_override,
            text=batch.text,
        )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment, text
    
    def configure_optimizers(self):
        """
        Configure optimizers for TTS and RL training.
        
        Returns separate optimizers for:
        1. TTS model (Tacotron2, GST, etc.)
        2. RL policy network (if RL training is enabled)
        """
        from torch.optim import Adam
        
        # TTS optimizer (all parameters except RL policy)
        tts_params = []
        rl_params = []
        
        for name, param in self.named_parameters():
            if 'rl_policy' in name:
                rl_params.append(param)
            else:
                tts_params.append(param)
        
        optimizers = []
        
        # TTS optimizer
        tts_optimizer = Adam(tts_params, lr=1e-3)
        optimizers.append(tts_optimizer)
        
        # RL policy optimizer (if RL training is enabled)
        if self.use_rl_training and len(rl_params) > 0:
            rl_optimizer = Adam(rl_params, lr=1e-4)
            optimizers.append(rl_optimizer)
        
        return optimizers[0] if len(optimizers) == 1 else optimizers
    
    def evaluate_trustworthiness(self, mel_spectrogram: Tensor, mel_len: Optional[Tensor] = None) -> Tensor:
        """
        Evaluate trustworthiness of generated mel spectrogram using HubERT.
        
        Note: HubERT expects waveform input, so this method should be called
        after converting mel spectrogram to waveform using a vocoder.
        
        Parameters
        ----------
        mel_spectrogram : Tensor
            Mel spectrogram of shape (batch, mel_channels, time)
        mel_len : Optional[Tensor]
            Lengths of mel spectrograms
            
        Returns
        -------
        Tensor
            Trustworthiness scores of shape (batch, 1)
        """
        if not self.use_hubert_classifier:
            raise ValueError("HubERT classifier is not enabled")
        
        # HubERT expects waveform input, not mel spectrogram
        # This method should be called after vocoder conversion
        # For now, return placeholder
        # TODO: Integrate vocoder and convert mel -> waveform before HubERT evaluation
        raise NotImplementedError(
            "Trustworthiness evaluation requires waveform input. "
            "Please convert mel spectrogram to waveform using a vocoder first."
        )
