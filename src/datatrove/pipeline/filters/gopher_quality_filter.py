import numpy as np

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.utils.typeshelper import Languages


STOP_WORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]


class GopherQualityFilter(BaseFilter):
    name = "🥇 Gopher Quality"

    def __init__(
        self,
        min_doc_words: int | None = 50,
        max_doc_words: int | None = 100000,
        min_avg_word_length: int | None = 3,
        max_avg_word_length: int | None = 10,
        max_symbol_word_ratio: float | None = 0.1,
        max_bullet_lines_ratio: float | None = 0.9,
        max_ellipsis_lines_ratio: float | None = 0.3,
        max_non_alpha_words_ratio: float | None = 0.8,
        min_stop_words: int | None = 2,
        stop_words: list[str] | None = None,
        exclusion_writer: DiskWriter = None,
        language: str = Languages.english,
    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        Args:
            min_doc_words:
            max_doc_words:
            min_avg_word_length:
            max_avg_word_length:
            max_symbol_word_ratio:
            max_bullet_lines_ratio:
            max_ellipsis_lines_ratio:
            max_non_alpha_words_ratio:
            min_stop_words:
            stop_words:
            exclusion_writer:
        """
        super().__init__(exclusion_writer)
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_word_ratio = max_symbol_word_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.max_non_alpha_words_ratio = max_non_alpha_words_ratio  # TODO rename to min_alpha_words_ratio
        self.min_stop_words = min_stop_words
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED


        Returns: False if sample.text does not pass any of the heuristic tests

        """
        text = doc.text
        words = split_into_words(text, self.language)
        n_words = len(words)

        n_non_symbol_words_words = 0
        non_symbol_chars = 0
        alpha_words = 0
        for word in words:
            has_non_symbol = False
            has_alpha = False
            for char in word:
                if char not in PUNCTUATION_SET:
                    has_non_symbol = True
                if char.isalpha():
                    has_alpha = True
                if has_non_symbol and has_alpha:
                    break
            if has_non_symbol:
                n_non_symbol_words_words += 1
                non_symbol_chars += len(word)
            if has_alpha:
                alpha_words += 1

        # words < min_doc_words or words > max_doc_words
        if self.min_doc_words and n_non_symbol_words_words < self.min_doc_words:
            return False, "gopher_short_doc"
        if self.max_doc_words and n_non_symbol_words_words > self.max_doc_words:
            return False, "gopher_long_doc"

        # mean word length is outside the range of 3 to 10 characters
        if n_non_symbol_words_words:
            avg_n_words = non_symbol_chars / n_non_symbol_words_words
        else:
            avg_n_words = np.mean([])
        if self.min_avg_word_length and avg_n_words < self.min_avg_word_length:
            return False, "gopher_below_avg_threshold"
        if self.max_avg_word_length and avg_n_words > self.max_avg_word_length:
            return False, "gopher_above_avg_threshold"

        # symbol-to-word ratio greater than 0.1 for either the hash symbol or the ellipsis
        if self.max_symbol_word_ratio:
            if text.count("#") / n_words > self.max_symbol_word_ratio:
                return False, "gopher_too_many_hashes"
            if (text.count("...") + text.count("…")) / n_words > self.max_symbol_word_ratio:
                return False, "gopher_too_many_ellipsis"

        # any document with more than 90 % of lines starting with a bullet point,
        # or more than 30 % ending with an ellipsis.
        lines = text.splitlines()
        if self.max_bullet_lines_ratio:
            bullet_lines = 0
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line.startswith("•") or stripped_line.startswith("-"):
                    bullet_lines += 1
            if bullet_lines / len(lines) > self.max_bullet_lines_ratio:
                return False, "gopher_too_many_bullets"
        if self.max_ellipsis_lines_ratio:
            ellipsis_lines = 0
            for line in lines:
                stripped_line = line.rstrip()
                if stripped_line.endswith("...") or stripped_line.endswith("…"):
                    ellipsis_lines += 1
            if ellipsis_lines / len(lines) > self.max_ellipsis_lines_ratio:
                return False, "gopher_too_many_end_ellipsis"

        # that 80 % of words in a document contain at least one alphabetic character
        if (
            self.max_non_alpha_words_ratio
            # nb of words with at least 1 alpha char < 0.8
            and alpha_words / n_words < self.max_non_alpha_words_ratio
        ):
            return False, "gopher_below_alpha_threshold"

        # stop word filter
        if self.min_stop_words and len(self.stop_words.intersection(set(words))) < self.min_stop_words:
            return False, "gopher_enough_stop_words"

        return True
