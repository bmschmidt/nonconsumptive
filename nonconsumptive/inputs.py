from pathlib import Path
from .document import Document
from typing import Callable, Iterator, Union, Optional, List, Tuple

class SingleFileFormat():
  __doc__ = """

  One text per line (no returns in any document).
  Id separated from the rest of the doucment by a tab.


  """
  def __init__(self,
    corpus,
    compression: Optional[str] = None,
    dir:Path = Path("input.txt"),
    format: str = "txt"):
    self.format = format
    self.corpus = corpus
    self.compression = compression
    self.dir = corpus.root / dir

  def documents(self) -> Iterator[Document]:
    # This is inefficient!
    for line in open(self):
      id, _ = doc.split("\t", 1)
      yield Document(self.corpus, id = id)

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    for line in open(self.corpus.root / self.dir):
      id, text = doc.split("\t", 1)
      yield id, text
      

class FolderInput():
  __doc__ = """
  Store files in folders, with optional compression.
  Can be nested at any depth.

  Ids are taken from filenames with ".txt.gz" removed.

  """
  def __init__(self,
    corpus,
    compression: Optional[str] = None,
    dir:Path = "texts",
    format: str = "txt"):
    self.format = format
    self.corpus = corpus
    self.compression = compression
    self.dir = corpus.root / dir

  def documents(self) -> Iterator[Document]:
    if self.compression is None:
      glob = f"**/*.{self.format}"
    else:
      glob = f"**/*.{format}.{compression}"
    assert(self.dir.exists())
    for f in self.dir.glob(glob):
      yield Document(self.corpus, path = f)

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    for doc in self.documents():
      id = doc.id
      yield id, doc.full_text
      