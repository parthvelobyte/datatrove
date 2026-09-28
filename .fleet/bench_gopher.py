import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), '.fleet', 'deps'))

from datatrove.data import Document
from datatrove.pipeline.filters.gopher_quality_filter import GopherQualityFilter
from datatrove.utils.word_tokenizers import WordTokenizer


class SimpleTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__('simple')

    def word_tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.replace('\n', ' ').split(' ') if tok]

    def sent_tokenize(self, text: str) -> list[str]:
        return [part.strip() for part in text.replace('?', '.').replace('!', '.').split('.') if part.strip()]

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        out = []
        pos = 0
        for sent in self.sent_tokenize(text):
            start = text.find(sent, pos)
            end = start + len(sent)
            out.append((start, end))
            pos = end
        return out


def make_docs():
    base = (
        'the quick brown fox jumps over the careful dog and that story will have context with useful words. '
        'clean natural language continues with enough alphabetic content to satisfy quality thresholds. '
        'another sentence with balanced average word lengths and no excessive punctuation.\n'
    )
    docs = []
    for i in range(2600):
        if i % 17 == 0:
            text = ('# ... !!! ' * 45) + base * 3
        elif i % 19 == 0:
            text = ('- bullet line with prose and context.\n' * 28) + base
        else:
            text = (base + f'unique marker {i} stable prose line.\n') * (4 + (i % 4))
        docs.append(Document(text=text, id=str(i)))
    return docs


def main():
    tok = SimpleTokenizer()
    quality = GopherQualityFilter(min_doc_words=10, max_doc_words=5000, language=tok)
    kept = 0
    reason_counts: dict[str, int] = {}
    total_chars = 0
    docs = make_docs()
    for _ in range(6):
        for doc in docs:
            probe = Document(text=doc.text, id=doc.id)
            result = quality.filter(probe)
            total_chars += len(probe.text)
            if result is True:
                kept += 1
            else:
                reason = result[1]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    parts = [f'{key}:{reason_counts[key]}' for key in sorted(reason_counts)]
    print(f'RESULT kept={kept} chars={total_chars} reasons=' + ','.join(parts))


if __name__ == '__main__':
    main()
