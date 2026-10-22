"""Deterministic sentence-preserving chunk construction with source offsets."""
from __future__ import annotations

from dataclasses import dataclass

import pysbd


@dataclass(frozen=True)
class ReadableChunk:
    text: str
    start: int
    end: int  # exclusive source offset
    token_count: int
    oversized_sentence: bool
    segmentation_fallback: bool = False


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sentence_preserving_chunks(
    paragraph: str,
    source_start: int,
    tokenizer,
    max_tokens: int = 240,
) -> list[ReadableChunk]:
    """Pack adjacent complete sentences up to ``max_tokens``.

    Paragraphs that fit remain intact. Longer paragraphs are split only at
    sentence boundaries. A sentence exceeding the unit cap remains intact and
    is explicitly flagged; it is never silently cut mid-sentence.
    Source offsets are half-open Unicode character offsets into the source text.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not paragraph:
        return []

    visible_paragraph = paragraph.strip()
    paragraph_tokens = _token_count(tokenizer, visible_paragraph)
    if paragraph_tokens <= max_tokens:
        return [ReadableChunk(
            text=visible_paragraph,
            start=source_start,
            end=source_start + len(paragraph),
            token_count=paragraph_tokens,
            oversized_sentence=False,
            segmentation_fallback=False,
        )] if visible_paragraph else []

    def retain_unsplit() -> list[ReadableChunk]:
        """Keep malformed segmenter output from silently corrupting offsets."""
        return [ReadableChunk(
            text=visible_paragraph,
            start=source_start,
            end=source_start + len(paragraph),
            token_count=paragraph_tokens,
            oversized_sentence=False,
            segmentation_fallback=True,
        )]

    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    spans = segmenter.segment(paragraph)
    if not spans:
        return retain_unsplit()

    # Keep any uncovered characters attached to adjacent text so chunks map to
    # a continuous source range rather than losing whitespace or punctuation.
    normalized: list[tuple[int, int]] = []
    cursor = 0
    for span in spans:
        start, end = int(span.start), int(span.end)
        if start < cursor or end <= start or end > len(paragraph):
            return retain_unsplit()
        if start > cursor:
            if normalized:
                normalized[-1] = (normalized[-1][0], start)
            else:
                start = cursor
        normalized.append((start, end))
        cursor = end
    if cursor < len(paragraph):
        normalized[-1] = (normalized[-1][0], len(paragraph))
    if normalized[0][0] != 0 or normalized[-1][1] != len(paragraph):
        raise ValueError("Sentence segmentation did not cover the paragraph")

    chunks: list[ReadableChunk] = []
    current_start: int | None = None
    current_end: int | None = None

    def emit(start: int, end: int) -> None:
        raw = paragraph[start:end]
        visible = raw.strip()
        if not visible:
            return
        count = _token_count(tokenizer, visible)
        # An over-cap chunk is permitted only when it is exactly one complete
        # sentence. The flag makes the budget-feasibility exception auditable.
        spans_inside = [(a, b) for a, b in normalized if a >= start and b <= end]
        oversized = count > max_tokens
        if oversized and len(spans_inside) != 1:
            raise AssertionError("Only a single complete sentence may exceed the unit cap")
        chunks.append(ReadableChunk(
            text=visible,
            start=source_start + start,
            end=source_start + end,
            token_count=count,
            oversized_sentence=oversized,
            segmentation_fallback=False,
        ))

    for sentence_start, sentence_end in normalized:
        sentence_text = paragraph[sentence_start:sentence_end].strip()
        if not sentence_text:
            continue
        if current_start is None:
            current_start, current_end = sentence_start, sentence_end
            continue
        combined_count = _token_count(tokenizer, paragraph[current_start:sentence_end].strip())
        if combined_count <= max_tokens:
            current_end = sentence_end
        else:
            emit(current_start, current_end)
            current_start, current_end = sentence_start, sentence_end

    if current_start is not None and current_end is not None:
        emit(current_start, current_end)

    return chunks
