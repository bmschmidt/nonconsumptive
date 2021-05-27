from pathlib import Path
from .document import Document
from typing import Callable, Iterator, Union, Optional, List, Tuple
import logging
import gzip
from pyarrow import feather, ipc
from .corpus import Corpus

logger = logging.getLogger("nonconsumptive")

class SingleFileInput():
  __doc__ = """

  One text per line (no returns in any document).
  Id separated from the rest of the doucment by a tab.

  """
  def __init__(self,
    corpus: Corpus,
    compression: Optional[str] = None,
    dir:Path = Path("input.txt"),
    format: str = "txt"):
    self.format = format
    self.corpus = corpus
    self.compression = compression
    self.dir = dir


  def __iter__(self) -> Iterator[Tuple[str, str]]:
    errored = []
    opener: function = open
    if self.compression == "gz":
      opener = lambda x: gzip.open(x, 'rt')
    print(opener)
    for line in opener(self.corpus.full_text_path):
      try:
        id, text = line.split("\t", 1)
        yield id, text
      except ValueError:
        errored.append(line)
    if len(errored):
      logger.warning(f"{len(errored)} unprintable lines, including:\n")
      logger.warning(*errored[:5])

class FolderInput():
  __doc__ = """
  Store files in folders, with optional compression.
  Can be nested at any depth.

  Ids are taken from filenames with ".txt.gz" removed.
  """
  def __init__(self,
    corpus: Corpus,
    compression: Optional[str] = None,
    format: str = "txt"):
    self.format = format
    self.corpus = corpus
    self.compression = compression
    self.dir = corpus.full_text_path

  def documents(self) -> Iterator[Document]:
    if self.compression is None:
      glob = f"**/*.{self.format}"
    else:
      glob = f"**/*.{self.format}.{self.compression}"
    assert(self.dir.exists())
    for f in self.dir.glob(glob):
      yield Document(self.corpus, path = f)

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    for doc in self.documents():
      id = doc.id
      yield id, doc.full_text
      

class MetadataInput():
  __doc__ = """
  For cases when the full text is stored inside the 
  metadata itself--as in common in, for example, social media datasets.

  Rather than parse multiple times, the best practice here is to convert to 
  feather *once* and then extract text and ids from the metadata column.
  """
  def __init__(self,
    corpus: Corpus):
    self.corpus = corpus
    self.text_field = corpus.text_field

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    fin = self.corpus.metadata.nc_metadata_path
    table = feather.read_table(fin, columns = ['@id', self.text_field])
    table = ipc.open_file(corpus.root / "nonconsumptive_catalog.feather")
    for i in range(table.num_record_batches):
      batch = table.get_batch(i)
      for id, text in zip(batch['@id'], batch[self.text_field]):
        yield id.as_py(), text.as_py()