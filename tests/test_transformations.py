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

class TestVolume():
    def test_can_make_dummy_corpus(self, simple_corpus):
        simple_corpus
    def test_wordcounts(self, simple_corpus):
        counter = simple_corpus.token_counts
        b = pa.record_batch([pa.array(['a','a', 'b'])], pa.schema({'token': pa.string()}))
        counted = counter.process_batch(b)
        assert counted.to_pandas()['count'].sum() == 3
        assert counted.to_pandas().query("token == 'a'")['count'].sum() == 2