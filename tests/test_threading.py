from nonconsumptive import Metadata, Corpus
from nonconsumptive.corpus import Bookstack
from nonconsumptive.metadata import *
import pytest
from pathlib import Path
import pandas as pd
import pyarrow as pa
from pyarrow import feather, json as pa_json
import json



@pytest.fixture(scope="session")
def corrected_dissertations(tmpdir_factory):
  fn = tmpdir_factory.mktemp("ndjson").join("catalog.ndjson")
  with open(fn, "w") as fout:
    for line in open("tests/catalogs/dissertations.ndjson"):
      d = json.loads(line)
      for k in [*d.keys()]:
        if d[k] == "NA":
          del d[k]
      fout.write(json.dumps(d) + "\n")
  return Path(str(fn))


@pytest.fixture(scope="function")
def dissertation_corpus(corrected_dissertations, tmpdir):
  return Corpus(texts = None,
            metadata = corrected_dissertations,
            dir = tmpdir,
            text_options = {"metadata_field" : "dissertation"})

@pytest.fixture(scope="function")
def non_metadata_corpus(tmpdir_factory):
  dir = Path(str(tmpdir_factory.mktemp("testing")))
  return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                metadata = None,
                dir = dir, text_options = {"format" : "txt"}, cache_set = {})  

class TestChunkPlan():
  def test_chunk_creation(self, dissertation_corpus):
    # Dissertation_corpus has 12 elements; ensure they're the
    # right length chunks
    dissertation_corpus._create_bookstack_plan(size = 4)
    feathers = [*dissertation_corpus.root.glob("bookstacks/*.feather")]
    assert len(feathers) == 3
    dissertation_corpus._create_bookstack_plan(size = 5)
    feathers = [*dissertation_corpus.root.glob("bookstacks/*.feather")]
    assert len(feathers) == 3
    for p in dissertation_corpus.root.glob("bookstacks/*.feather"):
      p.unlink()
    dissertation_corpus._create_bookstack_plan(size = 6)
    feathers = [*dissertation_corpus.root.glob("bookstacks/*.feather")]
    assert len(feathers) == 2

  def test_chunk_iteration(self, dissertation_corpus):
    dissertation_corpus._create_bookstack_plan(size = 4)
    tb = pa.Table.from_batches([*dissertation_corpus.tokenization()])
    assert len(tb) == 12


  def test_cached_chunk_iteration(self, dissertation_corpus):
    dissertation_corpus._create_bookstack_plan(size = 4)
    tb = pa.Table.from_batches([*dissertation_corpus.tokenization()])
    assert len(tb) == 12

    tb = pa.Table.from_batches([*dissertation_corpus.tokenization()])
    assert len(tb) == 12

class TestChunksSeparately():
  def test_chunk_instantiation(self, dissertation_corpus):
    d = dissertation_corpus._create_bookstack_plan(size = 4)
    stack1 = Bookstack(dissertation_corpus, "00001")
    assert len(stack1.ids) == 4

    tokenization = stack1.get_transform("tokenization")
    for t in tokenization:
      assert len(t) > 1
