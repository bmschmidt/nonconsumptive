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
    for batch in b.iter_docs():
        n += 1
        pass
    assert n == 3
      
  def test_tokenization_iterator_refreshes(self, simple_corpus):
    n = 0
    stack = simple_corpus.bookstacks[0]
    
    b = stack.get_transform("tokenization")

    for _ in b.iter_docs():
        n += 1
    assert (n == 3)

    for _ in b.iter_docs():
        n += 1
    assert (n == 6)

  def test_iterator_refreshes(self, simple_corpus):
    total = 0
    n = 0
    stack = simple_corpus.bookstacks[0]    
    for batch in stack.get_transform("token_counts"):
        total += batch['token_counts'].flatten().flatten()[1].to_numpy().sum()
    assert 42 <= total <= 43 # Different tokenizers produce slightly different results.

    total = 0
    for batch in stack.get_transform("token_counts"):
        total += batch['token_counts'].flatten().flatten()[1].to_numpy().sum()
    assert 42 <= total <= 43 # Different tokenizers produce slightly different results.


  def test_idlist_refreshes(self, simple_corpus):
      assert len([*simple_corpus.text_input.ids()]) == 3
      assert len([*simple_corpus.text_input.ids()]) == 3

class TestBookstacks():
  def test_chunk_instantiation(self, dissertation_corpus):
    stack1 = Bookstack(dissertation_corpus, "00000")
    tokenization = stack1.get_transform("tokenization")
    for tokens in tokenization:
        pass
    counts = stack1.get_transform("token_counts")
    for counts in counts:
        pass

class TestNgrams():
  def test_bigrams(self, dissertation_corpus):
    d = dissertation_corpus.metadata
    stack1 = Bookstack(dissertation_corpus, "00000")
    bigrams = stack1.get_transform("bigrams")
    bigrams = pa.Table.from_batches([*bigrams]).to_pandas()
  def test_encoded_bigrams(self, dissertation_corpus):
    d = dissertation_corpus.metadata
    stack1 = Bookstack(dissertation_corpus, "00000")
    bigrams = stack1.get_transform("encoded_bigrams")
    bigrams = pa.Table.from_batches([*bigrams]).to_pandas()


class TestSRP():
  def test_srp_instantiation(self, dissertation_corpus):
    d = dissertation_corpus.bookstacks[0]
    d.get_transform
    stack1 = Bookstack(dissertation_corpus, "00000")
    srp = stack1.get_transform("srp")
    for transformed in srp.iter_with_ids():
        pass
    # Haven't yet determined the plan here.
    transformed['SRP'][0]