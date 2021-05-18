from pathlib import Path
from .document import Document, tokenize, token_counts
import pyarrow as pa
from pyarrow import parquet, feather
from pyarrow import compute as pc
import json
import numpy as np
import logging
from bounter import bounter
from collections import defaultdict
from typing import DefaultDict, Iterator, Union, Optional, List
import polars as pl

from .prefs import prefs
from .metadata import Metadata
from .data_storage import ArrowReservoir
from .inputs import FolderInput, SingleFileFormat

# Custom type hints

_Path = Union[str, Path]


class Corpus():
  def __init__(self, dir, texts = None, 
                     metadata = None, cache_set = None, 
                     format = None, compression = None, 
                     text_input_method = None):
    """Create a corpus

    Arguments:
    dir -- the location to save derived files.
    texts -- the location of texts to ingest (a file or a folder.) If None, no text can be read
             but metadata functions may still work.
    metadata -- the location of a metadata file with a suffix indicating 
                the file type (.parquet, .ndjson, .csv, etc.) If not passed, bare-bones metadata will 
                be created based on filenames.
    cache_set -- Which elements created by the nc_pipeline should be persisted to disk. If None,
                 defaults will be taken from .prefs().
    format    -- The text format used ('.txt', '.html', '.tei', '.md', etc.)
    compression -- Compression used on texts. Only '.gz' currently supported.
    text_input_method -- deprecated?

    """
    if texts is not None: 
      self.full_text_path = Path(texts)
    else:
      self.full_text_path = None
    if metadata is not None:
      self.metadata_path = Path(metadata)
    else:
      self.metadata_path = None
    self.root: Path = Path(dir)
    self._metadata = None
    # which items to cache.
    self._cache_set = cache_set
    self.format = format
    if self.format is None:
      self.format = "txt"
    self.compression = compression
    self._text_input_method = text_input_method

  def setup_input_method(self, full_text_path, method):
    if method is not None:
      return method
    if full_text_path.is_dir():
      return 

  @property 
  def texts(self):
    return self.text_input_method(self, compression = self.compression, format = self.format)
    
  @property
  def text_input_method(self):
    # Defaults based on passed input. This may become the only method--
    # it's not clear to me why one should have to pass both.
    if self._text_input_method is None:
      if self.full_text_path is None:
        raise NotImplementedError("Streams not suppported. Must pass a file.")
      if self.full_text_path.is_dir():
        self._text_input_method = "folder"
      if self.full_text_path.is_file():
        self._text_input_method = "input.txt"
    if self._text_input_method == "folder":
      self.text_location = self.full_text_path
      assert self.text_location.exists()
      return FolderInput
    elif self._text_input_method == "input.txt":
      return SingleFileFormat

  def clean(self, targets = ['metadata', 'tokenization', 'token_counts']):
    import shutil
    for d in targets:
      shutil.rmtree(self.root / d)

  @property
  def metadata(self) -> Metadata:
    if self._metadata is not None:
      return self._metadata
    derived = Path(self.root / "metadata_derived.parquet")
    if derived.exists():
      self._metadata = Metadata.from_file(self, derived)
    try:
      mf = prefs('paths.metadata_file')
      print("\n\n", mf, "\n\n")
      self._metadata = Metadata.from_file(self, mf)
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
    p1 = self.full_text_path / (id + ".txt.gz")
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

  @property
  def total_wordcounts(self, max_bytes=100_000_000) -> pa.Table:
    cache_loc = self.root / "wordids.parquet"
    if cache_loc.exists():
      return parquet.read_table(cache_loc)
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
    if "wordids" in prefs('cache') or 'wordids' in self.cache_set:
      parquet.write_table(tb, cache_loc)
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
    w = self.total_wordcounts
    tokens = w['token']
    counts = w['count']
    return dict(zip(tokens.to_pylist(), counts.to_pylist()))

  @property
  def encoded_wordcounts(self):
    i = 0
    bookid_lookup = self.metadata.id_to_int_lookup
    wordids = pl.from_arrow(self.total_wordcounts.select(["token", "wordid"]))
    batches = []
    cache_size = 0

    for batch in self.token_counts:
        try:
            id = batch.schema.metadata.get(b"id").decode("utf-8")
            bookid = bookid_lookup[id]
        except KeyError:
            raise
        tab = pa.table({
            'bookid': pa.array(np.full(len(batch), bookid, dtype = np.uint32), pa.uint32()),
            'token': batch['token'],
            'count': batch['count']
        })
        batches.append(tab)
        cache_size += tab.nbytes
        if cache_size > 5_000_000:
          remainder = pl.from_arrow(pa.concat_tables(batches))
          q = wordids.join(remainder, left_on=["token"], right_on=["token"], how="inner", )
          for b in q.to_arrow().select(["bookid", "wordid", "count"]).to_batches():
              yield b          
          cache_size = 0
          batches = []
    remainder = pl.from_arrow(pa.concat_tables(batches))
    q = wordids.join(remainder, left_on=["token"], right_on=["token"], how="inner", )
    for b in q.to_arrow().select(["bookid", "wordid", "count"]).to_batches():
        yield b
  @property
  def bookid_wordcounts(self):
    bookid_lookup = self.metadata.id_to_int_lookup
    ids, counts = [], []
    cache_size = 0
    for batch in self.token_counts:
        try:
            id = batch.schema.metadata.get(b"id").decode("utf-8")
            bookid = bookid_lookup[id]
        except KeyError:
            raise
        count = pc.sum(batch['count']).as_py()
        ids.append(bookid)
        counts.append(count)
    return pa.table([pa.array(ids, pa.uint32()), pa.array(counts, pa.uint32())],
        names = ['bookid', 'nwords'])
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

class ChunkedTokenCounts(TokenCounts):
  name = "token_counts"
