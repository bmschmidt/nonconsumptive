import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
from nonconsumptive.corpus import Bookstack
import pyarrow as pa
from .test_metadata import dissertation_corpus, corrected_dissertations

@pytest.fixture(scope="function")
def simple_corpus(tmpdir_factory):
    dir = Path(str(tmpdir_factory.mktemp("testing")))
    return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                  metadata = None,
                  dir = dir, cache_set = {})

class TestDocument():
  def test_can_make_dummy_corpus(self, simple_corpus):
    simple_corpus

  def test_tokenization_iterator_iterates(self, simple_corpus):
    total = 0
    n = 0
    stack = simple_corpus.bookstacks[0]
    b = stack.get_transform("tokenization")
    for batch in b:
        n += 1
        pass
    assert n == 3
      
  def test_tokenization_iterator_refreshes(self, simple_corpus):
    n = 0
    stack = simple_corpus.bookstacks[0]
    
    b = stack.get_transform("tokenization")

    for _ in b:
        n += 1
    assert (n == 3)

    for _ in b:
        n += 1
    assert (n == 6)

  def test_iterator_refreshes(self, simple_corpus):
    total = 0
    n = 0
    stack = simple_corpus.bookstacks[0]    
    for batch in stack.get_transform("token_counts"):
        total += batch.to_pandas()['count'].sum()
        n += 1
        assert batch.to_pandas().shape[1] == 2
    assert n == 3
    assert 42 <= total <= 43 # Different tokenizers produce slightly different results.

    total = 0
    n = 0
    for batch in stack.get_transform("token_counts"):
        total += batch.to_pandas()['count'].sum()
        n += 1
        assert batch.to_pandas().shape[1] == 2
    assert n == 3
    assert 42 <= total <= 43 # Different tokenizers produce slightly different results.



  def test_idlist_refreshes(self, simple_corpus):
      assert len([*simple_corpus.text_input.ids()]) == 3
      assert len([*simple_corpus.text_input.ids()]) == 3

class TestBookstacks():
  def test_chunk_instantiation(self, dissertation_corpus):
    d = dissertation_corpus._create_bookstack_plan(size = 4)
    stack1 = Bookstack(dissertation_corpus, "00001")
    tokenization = stack1.get_transform("tokenization")
    for tokens in tokenization:
        pass
    counts = stack1.get_transform("token_counts")
    for counts in counts:
        pass
