from pathlib import Path
from .document import Document, tokenize, token_counts
import pyarrow as pa
from pyarrow import parquet
from pyarrow import feather
from pyarrow import compute as pc
import json
import yaml
import uuid
import numpy as np
from .prefs import prefs
import logging
from .metadata import Metadata
from .data_storage import ArrowReservoir
from .inputs import FolderInput, MalletInput
from bounter import bounter

from typing import Callable, Iterator, Union, Optional, List

# Custom type hints

_Path = Union[str, Path]

class Corpus():
  def __init__(self, dir, cache_set = None, format = None, compression = None, text_input_method = "folder"):
    self.root: Path = Path(dir)
    self._metadata = None
    self.text_location = self.root / prefs('paths.text_files')
    assert self.text_location.exists()
    # which items to cache.
    self._cache_set = cache_set
    self.format = format
    self.compression = compression
    self._text_input_method = text_input_method

  @property 
  def texts(self):
    return FolderInput(self, compression = self.compression, format = self.format)
  @property
  def text_input_method(self):
    if self._text_input_method == "folder":
      return FolderInput
    elif self._text_input_method == "mallet":
      return MalletInput

  @property
  def metadata(self) -> Metadata:
    print("here!!")

    if self._metadata is not None:
      print("here!!")
      return self._metadata
    derived = Path(self.root / "metadata_derived.parquet")
    if derived.exists():
      self._metadata = Metadata.from_file(self, derived)
    try:
      mf = prefs('paths.metadata_file')
      self._metadata: Metadata = Metadata.from_file(self, mf)
    except:
      self._metadata = Metadata.from_filenames(self)

    return self._metadata

  @property
  def cache_set(self):
    if self._cache_set:
      return self._cache_set
    else:
      return {}

  @property 
  def tokenization(self):
    """
    returns a reservoir that iterates over tokenized 
    arrow batches of documents as arrow arrays. 
    """
    tokenizer = Tokenization(self)
    return tokenizer

  @property
  def token_counts(self):
    """
    Returns a reservoir that iterates over grouped
    counts of tokens.
    """
    token_counts = TokenCounts(self)
    return token_counts

    

  def path_to(self, id):
    p1 = self.root / ("texts/" + id + ".txt.gz")
    if p1.exists():
      return p1
    logging.error(FileNotFoundError("HMM"))
    
  @property
  def documents(self):
    if self.metadata and self.metadata.tb:        
      for id in self.metadata.tb[self.metadata.id_field]:
        yield Document(self, str(id))

  def get_document(self, id):
    return Document(self, id)

  def total_wordcounts(self, max_bytes=100_000_000) -> pa.Table:
    counter = bounter(2048)
    for token, count in self.token_counts:
      dicto = dict(zip(token.to_pylist(), count.to_pylist()))
      counter.update(dicto)
    tokens, counts = zip(*counter.items())
    del counter
    tb = pa.table([pa.array(tokens, pa.utf8()), pa.array(counts, pa.uint32())],
      names = ['token', 'count']
    )
    tb = tb.take(pc.sort_indices(tb, sort_keys=[("count", "descending")]))
    tb = tb.append_column("wordid", pa.array(np.arange(len(tb)), pa.uint32()))
    return tb

  def feature_counts(self, dir = None):
    if dir is None:
      dir = self.root / "feature_counts"
    for f in Path(dir).glob("*.parquet"):
      metadata = parquet.ParquetFile(f).schema_arrow.metadata.get(b'nc_metadata')
      metadata = json.loads(metadata.decode('utf-8'))
      yield (metadata, parquet.read_table(f))


  @property
  def wordids(self):
    """
    Returns wordids in a dict format. 
    """
    w = self.total_wordcounts()
    tokens = w['token']
    counts = w['count']
    return dict(zip(tokens.to_pylist(), counts.to_pylist()))


  @property
  def encoded_wordcounts(self):
    bookids = self.metadata.id_to_int_lookup()
    wordids = self.wordids
    for f in self.token_counts:
      id = f.schema.metadata.get(b"id").decode("utf-8")
      ids = [wordids[token] for token in f['token'].to_pylist()]
      yield pa.table({
        'bookid': pa.array(np.full(len(ids), bookids[id], dtype = np.uint32)),
        'wordid': pa.array(ids, pa.uint32()),
        'count': f['count']
      })

  def write_feature_counts(self,
    single_files:bool = True,
    chunk_size:Optional[int] = None,
    batch_size:int = 250_000_000):
    
    """
    Write feature counts with metadata for all files in the document.

    dir: the location to place files into.
    chunking: whether to break up files into chunks of n words.
    single_files: Whether to write file per document, or group into batch
    batch_size bytes.
    """
    dir = self.root / Path(prefs("paths.feature_counts"))
    dir.mkdir(parents = True, exist_ok = True)
    bytes = 0
    counts = []
    batch_num = 0
    for doc in self.documents:
      try:
          wordcounts = doc.wordcounts
      except IndexError:
          print(doc.id)
          raise
      if single_files:
        parquet.write_table(pa.Table.from_batches([wordcounts]), (dir / f"{doc.id}.parquet").open("wb"))
      elif bytes > batch_size:
        counts.append(wordcounts)
        # Keep track of how big the table is.
        bytes += wordcounts.nbytes
        tab = pa.Table.from_batches(counts)
        feather.write_feather(tab, (dir / str(batch_num)).with_suffix("feather"))
        counts = []
        batch_num += 1
    if not single_files:
      # final flush
      tab = pa.Table.from_batches(counts)
      feather.write_feather(tab, (dir / str(batch_num)).with_suffix("feather"))
           
  def first(self) -> Document:
    for doc in self.documents:
      break
    return doc

  def random(self) -> Document:
    import random
    i = random.randint(0, len(self.metadata.tb) - 1)
    id = self.metadata.tb.column(self.metadata.id_field)[i].as_py()
    return Document(self, id)


class Tokenization(ArrowReservoir):

  name = "tokenization"

  arrow_schema = pa.schema({
    'token': pa.utf8()
  })

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    for id, text in self.corpus.texts:
      tokens = tokenize(text)
      yield pa.record_batch(
        [tokens],
        schema=self.arrow_schema.with_metadata({"id": id})
      )
"""
  def get_tokens(self, id):
    p = self.id_map[id]
    return parquet.read_table(p, columns=["token"], filters = [[("id", "=", id)]]).column("token")
  def get_word_usage(self, token):
    ds = parquet.ParquetDataset(self.path, filters = [[
      ['token', '=', token]
      ]], use_legacy_dataset=False)
    return ds.read(['id']).column("id")

  def ingest_function(self, id, val = None):
    pass
"""

class TokenCounts(ArrowReservoir):
  name = "token_counts"
  arrow_schema = {
    'token': pa.utf8(),
    'count': pa.uint32()
  }

  def _from_local(self):
    pass

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    for batch in self.corpus.tokenization:
      id = batch.schema.metadata.get(b"id").decode("utf-8")
      yield token_counts(batch['token'], id)

