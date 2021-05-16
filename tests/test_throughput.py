import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
import pyarrow as pa

@pytest.fixture(scope="module")
def simple_corpus():
    return Corpus(source_dir = Path('tests', 'corpora', 'test1'), target_dir = "tmp", format = "txt")

class TestVolume():
    def test_text_iteration(self, simple_corpus):
        ids = []
        texts = []
        for (id, text) in simple_corpus.texts:
            ids.append(id)
            texts.append(text)
        assert len(ids) == 3
        assert len(texts) == 3
        for id in ids:
            assert len(id) == 1
        assert sum(map(len, texts)) == 208
        assert min(map(len, texts)) == 0

    def test_tokenization_iteration(self, simple_corpus):
        ids = []
        texts = []
        for batch in simple_corpus.tokenization:
            ids.append(batch.schema.metadata.get(b'id').decode("utf-8"))
            texts.append(batch['token'].to_pylist())
        assert len(ids) == 3
        assert sum(map(len, texts)) == 42

    def test_wordcount_iteration(self, simple_corpus):
        ids = []
        words = []
        counts = 0
        for batch in simple_corpus.token_counts:
            ids.append(batch.schema.metadata.get(b'id').decode("utf-8"))
            counts += sum(batch['count'].to_pylist())
            words = words + batch['token'].to_pylist()
        assert len(ids) == 3
        assert "wife" in words
        assert counts == 42

    def test_total_wordcounts(self, simple_corpus):
        counts = simple_corpus.total_wordcounts
        # most common word should be 'a'
        assert counts.to_pandas()['token'][0] == 'a'
        df = counts.to_pandas()
        assert(counts['wordid'].to_pandas().max() == counts.shape[0] -1 )

    def test_bookids(self, simple_corpus):
        meta = simple_corpus.metadata
        lookup = meta.id_to_int_lookup
        for letter in "abг":
            assert letter in lookup

    def test_basic_wordids(self, simple_corpus):
        wordids = simple_corpus.wordids
        for word in ["wife", "fortune", "каждая"]:
            assert word in wordids

    def test_encode_wordcounts(self, simple_corpus):
        total = 0

        for batch in simple_corpus.encoded_wordcounts:
            print("GAHH", batch.to_pandas())
            total += batch.to_pandas()['count'].sum()
            assert batch.to_pandas().shape[1] == 3
        assert total == 42