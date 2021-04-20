from pathlib import Path
from .document import Document
from pyarrow import parquet
from pyarrow import feather
import pyarrow as pa
import json

from .metadata import Metadata
import gzip

from typing import Callable, Iterator, Union, Optional, List

_Path = Union[str, Path]

class Corpus():
  def __init__(self, metadata = None):
    pass

  def top_n_words(self, n = 1_000_000):
    pass

  
  def encode_feature_counts(self, dictionary: dict):
    """
    Encode feature counts as integers.
    """
    pass

  def documents(self) -> Iterator[Document]:
    raise NotImplementedError("No document method")
    pass
    """
    A corpus where each document is a full-text 
    """
  def feature_counts(self, dir):
    for f in Path(dir).glob("*.parquet"):
      metadata = parquet.ParquetFile(f).schema_arrow.metadata.get(b'nc_metadata')
      metadata = json.loads(metadata.decode('utf-8'))
      yield (metadata, parquet.read_table(f))

  def write_feature_counts(self,
    dir: Union[str, Path],
    single_files:bool = True,
    batch_size:int = 50_000_000):
    """
    Write feature counts with metadata for all files in the document.

    dir: the location to place files into.
    single_files: Whether to write file per document, or group into batch
    batch_size bytes.
    """
    dir = Path(dir)
    dir.mkdir(parents = True, exist_ok = True)
    bytes = 0
    counts = []
    batch_num = 0
    for doc in self.documents():
      wordcounts = doc.wordcounts
      if single_files:
        parquet.write_table(pa.Table.from_batches([wordcounts]), dir / f"{doc.id}.parquet")
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

class FolderCorpus(Corpus):
  def __init__(self, text: _Path, metadata: _Path, text_args = {}, metadata_args = {}):
    """
    text: location of the text to load.
    metadata: a metadata file path in a supported format. (.ndjson, .json, .parquet, .csv, MARC, etc.)

    text_args: passed to the textfile constructor.
    metadata_args: passed the the metadata constructor. (for instance: id_field.)
    """
    self.dir = Path(text)
    self.metadata = Metadata.from_file(Path(metadata))

  def first(self) -> Document:
    for doc in self.documents():
      break
    return doc


  def files(self) -> Iterator[Path]:
    for path in self.dir.glob("*.txt*"):
      yield path

  def documents(self) -> Iterator[Document]:
    """
    Iterator over the documents. Binds metadata to them as created.
    """
    for document in self.files():
      yield Document(self, path=document)
