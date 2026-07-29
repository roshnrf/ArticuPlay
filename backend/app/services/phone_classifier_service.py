import asyncio
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed.tensor  # noqa: F401 — must import before peft, see tasks/lessons.md
import torch.nn as nn
from peft import PeftModel
from transformers import WhisperFeatureExtractor, WhisperModel

from app.core.config import settings

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
WHISPER_SAMPLE_RATE = 16000


class PhoneClassifier(nn.Module):
    """Same architecture as research/phone_classifier/train_classifier_unfrozen.py —
    must match exactly, since we're loading those trained weights."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.d_model, 1)

    def forward(self, input_features):
        hidden = self.encoder(input_features).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.head(pooled).squeeze(-1)


@lru_cache(maxsize=1)
def _get_model() -> PhoneClassifier | None:
    """Loads the phone classifier once per process. Returns None if the model artifact
    isn't present (feature degrades gracefully — score() just skips the phone check)."""
    if not settings.PHONE_CLASSIFIER_PATH:
        return None
    model_dir = BACKEND_ROOT / settings.PHONE_CLASSIFIER_PATH
    if not model_dir.exists():
        return None

    base_encoder = WhisperModel.from_pretrained("openai/whisper-small").encoder
    lora_encoder = PeftModel.from_pretrained(base_encoder, str(model_dir / "encoder_lora"))
    model = PhoneClassifier(lora_encoder)
    model.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
    model.eval()
    return model


@lru_cache(maxsize=1)
def _get_feature_extractor() -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


class PhoneClassifierService:
    """Audio-based correct/needs-work signal for words containing a velar (/k/, /g/) or
    rhotic (/ɹ/) phoneme — trained on real disordered child speech (UltraSuite UXSSD,
    67.9% leave-one-speaker-out CV accuracy). Used as a one-directional check on top of
    ASR-transcript scoring: see session_service.score() for why (transcript scoring alone
    has a measured 15-28% false-accept rate on real disordered speech).

    Judges the raw audio directly — doesn't need the target word's text, only that it falls
    in the classifier's trained phone distribution (checked by the caller via
    app.utils.ipa.needs_phone_check before calling this)."""

    async def is_flagged_incorrect(self, audio_bytes: bytes) -> bool | None:
        """Returns True if the classifier thinks this attempt is wrong, False if it thinks
        it's fine, None if the model isn't available (never blocks the drill loop)."""
        return await asyncio.to_thread(self._classify_sync, audio_bytes)

    def _classify_sync(self, audio_bytes: bytes) -> bool | None:
        model = _get_model()
        if model is None:
            return None

        waveform = self._decode_to_pcm(audio_bytes)
        feature_extractor = _get_feature_extractor()
        inputs = feature_extractor(waveform, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logit = model(inputs.input_features)
            probability_correct = torch.sigmoid(logit).item()
        return probability_correct < 0.5

    @staticmethod
    def _decode_to_pcm(audio_bytes: bytes) -> np.ndarray:
        """Decodes arbitrary input audio (wav, webm/opus from the browser, mp3, ...) to
        16kHz mono float32 PCM via ffmpeg — same decode path faster-whisper uses
        internally, so we accept whatever format ASRService already accepts."""
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ar", str(WHISPER_SAMPLE_RATE), "-ac", "1", "pipe:1", "-loglevel", "error"],
            input=audio_bytes,
            capture_output=True,
            check=True,
        )
        return np.frombuffer(proc.stdout, dtype=np.float32)
