import edge_tts

VOICE_BY_LANGUAGE = {
    "en": "en-US-AnaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}


class TTSService:
    """Speaks a drill word via Microsoft edge-tts (free, asyncio-native — no thread offload needed)."""

    async def synthesize(self, word: str, language: str = "en") -> bytes:
        voice = VOICE_BY_LANGUAGE.get(language, VOICE_BY_LANGUAGE["en"])
        communicate = edge_tts.Communicate(word, voice)

        chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.extend(chunk["data"])
        return bytes(chunks)
