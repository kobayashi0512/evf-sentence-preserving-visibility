from __future__ import annotations

import unittest
from unittest.mock import patch

from src.sentence_chunking import sentence_preserving_chunks


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


class SentenceChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def test_sentence_boundaries_and_abbreviations_are_preserved(self) -> None:
        paragraph = (
            "Dr. Smith observed a response in 12 patients. "
            "However, no effect was detected in the control group. "
            "The result was reported in Fig. 2."
        )
        chunks = sentence_preserving_chunks(paragraph, 100, self.tokenizer, max_tokens=8)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            source_text = paragraph[chunk.start - 100:chunk.end - 100].strip()
            self.assertEqual(chunk.text, source_text)
            self.assertIn(chunk.text[-1], ".?!")

    def test_within_budget_paragraph_is_left_intact(self) -> None:
        paragraph = "First sentence. Second sentence."
        chunks = sentence_preserving_chunks(paragraph, 7, self.tokenizer, max_tokens=4)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, paragraph)
        self.assertEqual((chunks[0].start, chunks[0].end), (7, 7 + len(paragraph)))

    def test_long_sentence_is_not_split(self) -> None:
        sentence = "One two three four five six seven eight nine."
        chunks = sentence_preserving_chunks(sentence, 0, self.tokenizer, max_tokens=4)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].oversized_sentence)
        self.assertEqual(chunks[0].text, sentence)

    def test_unicode_offsets_map_back_to_original_text(self) -> None:
        paragraph = "μg dosage was tested. Résults were stable."
        chunks = sentence_preserving_chunks(paragraph, 20, self.tokenizer, max_tokens=3)
        for chunk in chunks:
            self.assertEqual(paragraph[chunk.start - 20:chunk.end - 20].strip(), chunk.text)

    def test_invalid_overlapping_segmenter_offsets_retain_paragraph_and_flag_fallback(self) -> None:
        class Span:
            def __init__(self, start: int, end: int) -> None:
                self.start = start
                self.end = end

        class BadSegmenter:
            def __init__(self, **_: object) -> None:
                pass

            def segment(self, _: str) -> list[Span]:
                return [Span(0, 15), Span(14, 30)]

        paragraph = "First sentence. Second sentence."
        with patch("src.sentence_chunking.pysbd.Segmenter", BadSegmenter):
            chunks = sentence_preserving_chunks(paragraph, 40, self.tokenizer, max_tokens=3)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, paragraph)
        self.assertEqual((chunks[0].start, chunks[0].end), (40, 40 + len(paragraph)))
        self.assertTrue(chunks[0].segmentation_fallback)
        self.assertFalse(chunks[0].oversized_sentence)


if __name__ == "__main__":
    unittest.main()
