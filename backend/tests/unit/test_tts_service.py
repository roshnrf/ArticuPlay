import pytest

from app.services import tts_service
from app.services.tts_service import VOICE_BY_LANGUAGE, TTSService


class _FakeCommunicate:
    """Stands in for edge_tts.Communicate — real network calls don't belong in a unit test."""

    last_voice: str | None = None

    def __init__(self, text: str, voice: str):
        _FakeCommunicate.last_voice = voice

    async def stream(self):
        yield {"type": "audio", "data": b"chunk1"}
        yield {"type": "not-audio", "data": b"ignore-me"}
        yield {"type": "audio", "data": b"chunk2"}


@pytest.fixture(autouse=True)
def fake_edge_tts(monkeypatch):
    monkeypatch.setattr(tts_service.edge_tts, "Communicate", _FakeCommunicate)


@pytest.mark.asyncio
async def test_synthesize_concatenates_audio_chunks_only():
    result = await TTSService().synthesize("rabbit", language="en")
    assert result == b"chunk1chunk2"


@pytest.mark.asyncio
async def test_synthesize_picks_voice_by_language():
    await TTSService().synthesize("gato", language="ar")
    assert _FakeCommunicate.last_voice == VOICE_BY_LANGUAGE["ar"]


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_english_voice_for_unknown_language():
    await TTSService().synthesize("word", language="xx")
    assert _FakeCommunicate.last_voice == VOICE_BY_LANGUAGE["en"]
