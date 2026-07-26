import pytest

from app.services import asr_service
from app.services.asr_service import ASRService, _get_model


class _FakeSegment:
    def __init__(self, text, words=None):
        self.text = text
        self.words = words


class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _FakeInfo:
    language = "en"


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel — loading the real model is far too
    slow/heavy for a unit test, and we only need to verify our own wiring, not Whisper itself."""

    last_call_kwargs: dict = {}
    init_count = 0

    def __init__(self, *args, **kwargs):
        _FakeWhisperModel.init_count += 1

    def transcribe(self, audio, **kwargs):
        _FakeWhisperModel.last_call_kwargs = kwargs
        segments = [_FakeSegment(" rabbit ", words=[_FakeWord(" rabbit", 0.0, 0.5)])]
        return segments, _FakeInfo()


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    _get_model.cache_clear()
    _FakeWhisperModel.init_count = 0
    monkeypatch.setattr(asr_service, "WhisperModel", _FakeWhisperModel)
    yield
    _get_model.cache_clear()


@pytest.mark.asyncio
async def test_transcribe_assembles_transcript_and_words():
    result = await ASRService().transcribe(b"fake-audio-bytes", language="en")
    assert result.transcript == "rabbit"
    assert result.language == "en"
    assert result.words == [{"word": " rabbit", "start": 0.0, "end": 0.5}]


@pytest.mark.asyncio
async def test_transcribe_passes_target_word_as_initial_prompt():
    await ASRService().transcribe(b"fake-audio-bytes", language="en", target_word="rabbit")
    assert _FakeWhisperModel.last_call_kwargs["initial_prompt"] == "rabbit"


@pytest.mark.asyncio
async def test_transcribe_pins_temperature_to_zero_for_determinism():
    await ASRService().transcribe(b"fake-audio-bytes")
    assert _FakeWhisperModel.last_call_kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_model_is_loaded_once_across_multiple_transcriptions():
    await ASRService().transcribe(b"fake-audio-bytes")
    await ASRService().transcribe(b"fake-audio-bytes")
    await ASRService().transcribe(b"fake-audio-bytes")
    assert _FakeWhisperModel.init_count == 1
