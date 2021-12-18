import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
import pyarrow as pa
from pyarrow import compute as pc

@pytest.fixture(scope="function")
def simple_corpus(tmpdir_factory):
    dir = Path(str(tmpdir_factory.mktemp("testing")))
    return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                  metadata = None,   dir = dir, cache_set = {})


class TestTables():
    def test_documents_table(self, simple_corpus):
        p = simple_corpus.table("text")        
        assert len(p) == 3
        assert p['text']

    def test_tokenization_table(self, simple_corpus):
        tb = simple_corpus.table("tokenization")   
        assert len(tb) == 3
        assert 40 < len(pc.list_flatten(tb['tokenization'])) < 44
    def test_tokencounts_table(self, simple_corpus):
        tb = simple_corpus.table("token_counts")   
        assert len(tb) == 3
        as_rows = tb.to_pandas()['token_counts'].explode()
        total = as_rows.apply(lambda x: x['count']).sum()

        assert 40 <= total <= 44

class TestIteration():
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
            texts.append(batch['tokenization'].values.to_pylist())            
        total = sum(map(len, texts))
        assert 42 <= total <= 43

    def test_wordcount_iteration(self, simple_corpus):
        words = []
        total = 0
        tb = pa.Table.from_batches([*simple_corpus.token_counts()])
        words, counts = tb['token_counts'].combine_chunks().flatten().flatten()
        total = pc.sum(counts).as_py()
        words = words.to_pylist()
        assert "wife" in words
        assert "каждая" in words
        assert 42 <= total <= 43

    def test_document_lengths(self, simple_corpus):
        length = 0
        tb = pa.Table.from_batches([*simple_corpus.document_lengths()])
        length = pc.sum(tb['nwords']).as_py()
        assert 42 <= length <= 43

    def test_total_wordcounts(self, simple_corpus):
        counts = simple_corpus.total_wordcounts
        # most common word should be 'a'; using pyarrow re2 it may be '', though.
        assert counts.to_pandas()['token'][0] in ['a', '']
        df = counts.to_pandas()
        assert(counts['wordid'].to_pandas().max() == counts.shape[0] - 1 )

    def test_ncids(self, simple_corpus):
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
        tb = pa.Table.from_batches([*simple_corpus.tokenization()])
        assert len(tb) == 3
        tb = pa.Table.from_batches([*simple_corpus.tokenization()])
        assert len(tb) == 3

    def test_encode_wordcounts(self, simple_corpus):
        total = 0
        n = 0
        for batch in simple_corpus.iter_over("encoded_unigrams"):
            total += batch.to_pandas()['count'].sum()
            n += 1
            assert batch.to_pandas().shape[1] == 3
        assert n == 1
        assert 42 <= total <= 43 # Different tokenizers produce slightly different results.