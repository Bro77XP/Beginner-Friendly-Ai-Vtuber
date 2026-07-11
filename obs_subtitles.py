"""
obs_subtitles.py

OBS subtitle system extracted from the original project.

Usage:

    from obs_subtitles import OBSSubtitles

    subtitles = OBSSubtitles()

    # Before playback starts
    subtitles.begin(response_text)

    sd.play(audio, sample_rate)

    # During your playback loop
    while sd.get_stream().active:
        subtitles.update()
        ...

    # After playback finishes
    subtitles.finish()
"""

import re
import threading
import time


class OBSSubtitles:
    def __init__(
        self,
        subtitle_file="obs_subtitles.txt",
        clear_delay=2.0,
        words_per_second=2.5,
        switch_early=4.2,
    ):
        self.subtitle_file = subtitle_file
        self.clear_delay = clear_delay
        self.words_per_second = words_per_second
        self.switch_early = switch_early

        self._chunks = []
        self._durations = []
        self._current_chunk = 0
        self._start_time = None

    # -------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------

    @staticmethod
    def strip_paralinguistic_tags(text: str) -> str:
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\*[^*]+\*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -------------------------------------------------------
    # File Writing
    # -------------------------------------------------------

    def write_subtitle(self, text: str):
        text = self.strip_paralinguistic_tags(text)

        with open(self.subtitle_file, "w", encoding="utf-8") as f:
            f.write(text)

    def clear_subtitle(self):
        with open(self.subtitle_file, "w", encoding="utf-8") as f:
            f.write("")

    # -------------------------------------------------------
    # Chunking
    # -------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, max_words=50):
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current = []
        words = 0

        for sentence in sentences:
            wc = len(sentence.split())

            if current and words + wc > max_words:
                chunks.append(" ".join(current))
                current = []
                words = 0

            current.append(sentence)
            words += wc

        if current:
            chunks.append(" ".join(current))

        return chunks

    # -------------------------------------------------------
    # Playback Sync
    # -------------------------------------------------------

    def begin(self, response: str):
        self._chunks = self.chunk_text(response)

        if not self._chunks:
            self._chunks = [response]

        self._durations = [
            len(chunk.split()) / self.words_per_second
            for chunk in self._chunks
        ]

        self._current_chunk = 0
        self._start_time = time.time()

        self.write_subtitle(self._chunks[0])

    def update(self):
        if self._start_time is None:
            return

        if self._current_chunk >= len(self._chunks) - 1:
            return

        elapsed = time.time() - self._start_time

        cumulative = sum(
            self._durations[: self._current_chunk + 1]
        )

        # Same timing logic as the original project
        if elapsed >= cumulative - self.switch_early:
            self._current_chunk += 1
            self.write_subtitle(self._chunks[self._current_chunk])

    def finish(self):
        threading.Timer(
            self.clear_delay,
            self.clear_subtitle,
        ).start()
