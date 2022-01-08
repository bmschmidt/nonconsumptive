from nonconsumptive.writers import ParquetWriter
from nonconsumptive import Corpus
from pyarrow import parquet
from pathlib import Path
import pytest

@pytest.fixture(scope="function")
def simple_corpus(tmpdir_factory):
    dir = Path(str(tmpdir_factory.mktemp("testing")))
    return Corpus(texts = Path('tests', 'corpora', 'test1', 'texts'), 
                  metadata = None,
                  dir = dir, cache_set = {})

class TestWriters():
  def test_bookstack_writer(self, simple_corpus, tmpdir_factory):
    testing = Path(str(tmpdir_factory.mktemp("testing")))
    ParquetWriter(simple_corpus, testing).write()
    texts = []
    ids = []
    for file in testing.glob("*.parquet"):
      p = parquet.read_table(file)
      ids += p['@id'].to_pylist()
      texts += p["nc:text"].to_pylist()
    assert len(texts) == 3
    assert len(ids) == 3
    pride = ids.index("a")
    assert "of a wife" in texts[pride]