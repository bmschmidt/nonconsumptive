import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
from nonconsumptive.inputs import FolderInput, MetadataInput, SingleFileInput
import pyarrow as pa
import json

@pytest.fixture(scope="module")
def three_files(tmpdir_factory):
    return FolderInput(Path("tests/corpora/test1/texts"))
    
@pytest.fixture(scope="module")
def single_file_format(tmpdir_factory):
  dest = tmpdir_factory.mktemp("testing")
  filedest = Path(str(dest.join("input.txt")))
  with open(filedest, "w") as fout:
    for p in Path("tests/corpora/test1/texts").glob("*.txt"):
      fout.write(f"{p.with_suffix('').name}\t{' '.join(p.open().readlines())}\n")
  corp = Corpus(dest, texts = filedest)
  return SingleFileInput(filedest, corpus = corp)

@pytest.fixture(scope="module")
def SOTU_corpus(tmpdir_factory):
  dest = Path(str(tmpdir_factory.mktemp("SOTU")))
  return Corpus(dest, bookstacks = Path("tests/corpora/SOTU"))

@pytest.fixture(scope="module")
def ndjson_format(tmpdir_factory):
  dest = tmpdir_factory.mktemp("testing")
  filedest = Path(str(dest.join("jsoncatalog.txt")))
  with open(filedest, "w") as fout:
    for p in Path("tests/corpora/test1/texts").glob("*.txt"):
      id = p.with_suffix('').name
      text = p.open().read()
      fout.write(json.dumps({
        '@id': id,
        'text': text
      }) + "\n")
  corp = Corpus(dest, metadata = filedest, text_options={"metadata_field": "text"})
  return MetadataInput(metadata = filedest, corpus = corp, metadata_field = "text")

class TestMetadataInput:
  def test_iter(self, ndjson_format):
    i = 0
    for t in ndjson_format:
      i += 1
    assert i == 3

  def test_get(self, ndjson_format):
    pride = ndjson_format['a']
    assert "of a wife" in pride
    
  def test_ids(self, ndjson_format):
    ids = [*ndjson_format.ids()]
    assert len(set(ids)) == 3

class TestFolderInput:
  def test_iter(self, three_files):
    i = 0
    for t in three_files:
      i += 1
    assert i == 3
  def test_get(self, three_files):
    pride = three_files['a']
    assert "of a wife" in pride
    
  def test_ids(self, three_files):
    ids = [*three_files.ids()]
    assert len(set(ids)) == 3

class TestFileInput:
  def test_iter(self, single_file_format):
    i = 0
    for t in single_file_format:
      i += 1
    assert i == 3
  
  def test_get(self, single_file_format):
    pride = single_file_format['a']
    assert "of a wife" in pride
    
  def test_ids(self, single_file_format):
    ids = [*single_file_format.ids()]
    assert len(set(ids)) == 3

class TestBookstacks():
  def test_bookstacks_creation(self, SOTU_corpus):
    [*SOTU_corpus.iter_over("tokenization")]

  def test_wordcounts(self, SOTU_corpus):
    a = SOTU_corpus.cache("unigrams")
    b = SOTU_corpus.cache("encoded_unigrams")
    m = [*SOTU_corpus.iter_over("bigrams")]
    assert(len(pa.Table.from_batches(m)) == 100)