import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), '.fleet', 'deps'))

from datatrove.data import Document
from datatrove.pipeline.filters.gopher_quality_filter import GopherQualityFilter
from datatrove.pipeline.filters.gopher_repetition_filter import GopherRepetitionFilter
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


def make_doc(i: int, noisy: bool = False) -> Document:
    base = (
        'the quick brown fox jumps over the careful dog and that story will have context with useful words. '
        'clean natural language continues with enough alphabetic content to satisfy quality thresholds. '
    )
    if noisy:
        text = ('# ... !!! ' * 80) + base
    else:
        text = (base + f'unique marker {i} with ordinary prose and stable endings.\n') * 8
    return Document(text=text, id=str(i))


def main():
    tok = SimpleTokenizer()
    quality = GopherQualityFilter(min_doc_words=10, max_doc_words=5000, language=tok)
    repetition = GopherRepetitionFilter(language=tok)
    good = make_doc(1)
    bad = make_doc(2, noisy=True)
    assert quality.filter(good) is True
    q_result = quality.filter(bad)
    assert isinstance(q_result, tuple) and q_result[0] is False
    rep_result = repetition.filter(Document(text=('repeat this exact line.\n' * 10), id='r'))
    assert isinstance(rep_result, tuple) and rep_result[0] is False
    print('RESULT check ok')


if __name__ == '__main__':
    main()
