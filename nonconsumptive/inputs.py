from pathlib import Path
from .document import Document
from typing import Callable, Iterator, Union, Optional, List, Tuple
import logging

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
    self.dir = dir

#  def documents(self) -> Iterator[Document]:
#    # This is inefficient!
#    for line in open(self):
#      try:
#        id, _ = line.split("\t", 1)
#        yield Document(self.corpus, id = id)
#      except IndexError:

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    errored = []

    for line in open(self.corpus.full_text_path):
      try:
        id, text = line.split("\t", 1)
        yield id, text
      except ValueError:
        errored.append(line)
    if len(errored):
      logging.warning(f"{len(errored)} unprintable lines, including:\n")
      logging.warning(*errored[:5])

class FolderInput():
  __doc__ = """
  Store files in folders, with optional compression.
  Can be nested at any depth.

  Ids are taken from filenames with ".txt.gz" removed.
  """
  def __init__(self,
    corpus,
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
      