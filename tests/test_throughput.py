import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
import pyarrow as pa

@pytest.fixture(scope="function")
def simple_corpus(tmpdir_factory):
    dir = Path(str(tmpdir_factory.mktemp("testing")))
    return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                  metadata = None,
                  dir = dir, cache_set = {})

class TestCorpus():
    def test_text_iteration(self, simple_corpus):
        ids = []
        texts = []
        for (text) in simple_corpus.text_input:
            texts.append(text)
        assert len(texts) == 3
        assert sum(map(len, texts)) == 208
        assert min(map(len, texts)) == 0

    def test_tokenization_iteration(self, simple_corpus):
        ids = []
        texts = []
        for batch in simple_corpus.tokenization():
            texts.append(batch['token'].to_pylist())
        total = sum(map(len, texts))
        assert 42 <= total <= 43

    def test_wordcount_iteration(self, simple_corpus):
        words = []
        total = 0
        for batch in simple_corpus.token_counts():
            total += sum(batch['count'].to_pylist())
            words = words + batch['token'].to_pylist()
        assert "wife" in words
        assert "каждая" in words
        assert 42 <= total <= 43

    def test_document_lengths(self, simple_corpus):
        lengths = simple_corpus.document_lengths()
        tab = pa.Table.from_batches([*lengths])
        assert 42 <= tab.to_pandas()['nwords'].sum() <= 43

    def test_total_wordcounts(self, simple_corpus):
        counts = simple_corpus.total_wordcounts
        # most common word should be 'a'
        assert counts.to_pandas()['token'][0] == 'a'
        df = counts.to_pandas()
        assert(counts['wordid'].to_pandas().max() == counts.shape[0] - 1 )

    def test__ncids(self, simple_corpus):
        meta = simple_corpus.metadata
        lookup = meta.ids.to_pylist()
        for letter in "abг":
            assert letter in lookup

    def test_basic_wordids(self, simple_corpus):
        wordids = simple_corpus.wordids
        for word in ["wife", "fortune", "каждая"]:
            assert word in wordids

    def test_text_input_refreshes(self, simple_corpus):
        total = 0
        n = 0
        for batch in simple_corpus.text_input:
            n += 1
        assert n == 3
        n = 0
        for batch in simple_corpus.text_input:
            n += 1
        assert n == 3

    def test_iterator_refreshes(self, simple_corpus):
        total = 0
        n = 0
        for batch in simple_corpus.tokenization():
            n += 1
        assert n == 3
        n = 0
        for batch in simple_corpus.tokenization():
            n += 1
        assert n == 3

    def test_encode_wordcounts(self, simple_corpus):
        total = 0
        n = 0
        for batch in simple_corpus.encoded_wordcounts():
            total += batch.to_pandas()['count'].sum()
            n += 1
            assert batch.to_pandas().shape[1] == 3
        assert n == 3
        assert 42 <= total <= 43 # Different tokenizers produce slightly different results.