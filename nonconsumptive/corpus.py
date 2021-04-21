from pathlib import Path
from .document import Document
from pyarrow import parquet
from pyarrow import feather
import pyarrow as pa
import json
import yaml

from .metadata import Metadata
import gzip

from typing import Callable, Iterator, Union, Optional, List

# Custom type hints

_Path = Union[str, Path]

class Corpus():
  def __init__(self, dir, prefs = None):
    self.root = Path(dir)
    if prefs is not None:
      self.prefs = prefs
    else:
      self.load_prefs()
    self.metadata = Metadata.from_file(self, self.prefs['paths']['metadata_file'])
    parquet.write_table(self.metadata.tb, self.root / "metadata_derived.parquet")
    self.text_location = self.root / self.prefs['paths']['text_files']
  
  def load_prefs(self):
     dir = self.root
     if (dir / "nc.yaml").exists():
         self.prefs = yaml.safe_load((dir / "nc.yaml").open())
     else:
         self.prefs = {
            "paths": {
              "text_files": "texts",
              "metadata_file": None,
              "feature_counts": "feature_counts",
              "SRP": None
            }
         }
     
  def top_n_words(self, n = 1_000_000):
    pass

  def encode_feature_counts(self, dictionary: dict):
    """
    Encode feature counts as integers.
    """
    pass

  def documents(self):
      # Just an alias for now.
      for document in self.documents_from_files():
          yield document
    
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
    batch_size:int = 50_000_000):
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
    print(self.text_location)
    for path in self.text_location.glob("*.txt*"):
      yield path

  def documents_from_files(self) -> Iterator[Document]:
    """
    Iterator over the documents. Binds metadata to them as created.
    """
    for document in self.files():
      yield Document(self, path=document)
