from pathlib import Path
from .document import Document
from pyarrow import parquet
from pyarrow import feather
import pyarrow as pa
import json
import yaml
import uuid
import numpy as np
from .prefs import prefs
import logging
from .metadata import Metadata
import gzip

from typing import Callable, Iterator, Union, Optional, List

# Custom type hints

_Path = Union[str, Path]

class Corpus():
  def __init__(self, dir):
    self.root: Path = Path(dir)
    mf = prefs('paths.metadata_file')
    self.metadata: Metadata = Metadata.from_file(self, mf)
    parquet.write_table(self.metadata.tb, self.root / "metadata_derived.parquet")
    self.text_location = self.root / prefs('paths.text_files')

  def top_n_words(self, n = 1_000_000):
    pass

  @property 
  def tokenization(self):

    tokenizer = Tokenization(self)
    if tokenizer.empty():
      tokenizer.create()
    return tokenizer

  def encode_feature_counts(self, dictionary: dict):
    """
    Encode feature counts as integers.
    """
    pass

  def path_to(self, id):
    p1 = self.root / ("texts/" + id + ".txt.gz")
    if p1.exists():
      return p1
    raise FileNotFoundError("HMM")

  def documents(self):
      # Just an alias for now.
      for document in self.documents_from_files():
          yield document

  def get_document(self, id):
    return Document(self, id)

  def feature_counts(self, dir = None):
    if dir is None:
      dir = self.root / "feature_counts"
    for f in Path(dir).glob("*.parquet"):
      metadata = parquet.ParquetFile(f).schema_arrow.metadata.get(b'nc_metadata')
      metadata = json.loads(metadata.decode('utf-8'))
      yield (metadata, parquet.read_table(f))

  def write_feature_counts(self,
    dir: Union[str, Path],
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
    dir = Path(dir)
    dir.mkdir(parents = True, exist_ok = True)
    bytes = 0
    counts = []
    batch_num = 0
    for doc in self.documents():
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
    for doc in self.documents():
      break
    return doc

  def files(self) -> Iterator[Path]:
    for path in self.text_location.glob("*.txt*"):
      yield path

  def documents_from_metadata(self) -> Iterator[Document]:
    pass

  def documents_from_files(self) -> Iterator[Document]:
    """
    Iterator over the documents. Binds metadata to them as created.
    """
    for document in self.files():
      yield Document(self, path=document)

class Tokenization():

  def __init__(self, corpus: Corpus, path: Optional[Path] = None):
    self.corpus = corpus
    if path is None:
      path = corpus.root / "tokenized"
    path.mkdir(exist_ok=True)
    self.path = path

  def clean(self):
    for p in self.path.glob("*.parquet"):
      p.unlink()

  def empty(self):
    for file in self.path.glob("*.parquet"):
      return False
    return True

  def create(self, max_size = 250_000_000):
    self.queue = []
    queue_size = 0
    for document in self.corpus.documents():
      tokens = document.tokenize()
      # Could speed up.Tokenization
      ids = pa.array([document.id] * len(tokens), pa.utf8())
      batch = pa.RecordBatch.from_arrays(
        [
          ids,
          tokens
        ],
        schema = pa.schema({
          'id': pa.utf8(),
          'token': pa.utf8()
        })
      )
      self.queue.append((document.id, batch))
      queue_size += batch.nbytes
      if queue_size > max_size:
        self.flush() # Intermedia write
        queue_size = 0
    self.flush() # final flush.

  def flush(self):
    if len(self.queue) == 0:
      return
    logging.warn(f"Flushing {len(self.queue)}")
    # Get it alphabetically by id
    self.queue.sort()
    tab = pa.Table.from_batches([q[1] for q in self.queue])
    fn = self.path / (str(uuid.uuid1()) + ".parquet")
    parquet.write_table(tab, fn)
    self.queue = []
  
  def get_tokens(self, id):
    print(self.path)
    ds = parquet.ParquetDataset(self.path, filters = [[
      ['id', '=', id]
      ]], use_legacy_dataset=False)
    return ds.read(['token']).column("token")

