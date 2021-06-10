from pathlib import Path
from .document import Document
from typing import Callable, Iterator, Union, Optional, List, Tuple
import logging
import gzip
from pyarrow import feather, ipc
from .data_storage import Node, BatchIDIndex
logger = logging.getLogger("nonconsumptive")

class TextInput(Node):
  """
  An object that defines a method for iterating across documents
  linked to IDs.

  There's one really important rule here:
  IT MUST BE IN A STABLE ORDER ACROSS MULTIPLE RUNS WITH THE SAME UUID!
  AND I DON'T KNOW HOW TO ENFORCE THIS!
  """
  def __init__(self, corpus):
    super().__init__(corpus)

  def _iter_documents(self):
    raise NotImplementedError("No documents method defined.")

  def __getitem__(self):
    raise NotImplementedError("This text input type does not support random item access.")

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    """
    The texts iterate over the documents in order
    and stash the ids somewhere.
    """
    if self.corpus._metadata is not None:
      with BatchIDIndex(self.corpus) as batch_ids:
        create_ids = not batch_ids.exists()
        for id, text in self._iter_documents():
          if create_ids:
            batch_ids.push(id)
          yield id, text
    else:
      for id, text in self._iter_documents():
        yield id, text

class SingleFileInput(TextInput):
  __doc__ = """

  One text per line (no returns in any document).
  id field separated from the rest of the doucment by a tab. (no tabs allowed in id.)

  """
  def __init__(self,
        corpus: 'Corpus',
        compression: Optional[str] = None,
        dir:Path = Path("input.txt"),
        format: str = "txt"):

    self.format = format
    self.compression = compression
    self.dir = dir
    super().__init__(corpus)

  def _iter_documents(self) -> Iterator[Tuple[str, str]]:
    errored = []
    opener: function = open
    if self.compression == "gz":
      opener = lambda x: gzip.open(x, 'rt')
    for line in opener(self.corpus.full_text_path):
      try:
        id, text = line.split("\t", 1)
        yield id, text
      except ValueError:
        errored.append(line)
    if len(errored):
      logger.warning(f"{len(errored)} unprintable lines, including:\n")
      logger.warning(*errored[:5])

class FolderInput(TextInput):
  __doc__ = """
  Store files in folders, with optional compression.
  Can be nested at any depth.

  Ids are taken from filenames with ".txt.gz" removed.
  """
  def __init__(self,
        corpus: "Corpus",
        compression: Optional[str] = None):
    self.format = corpus.format
    self.corpus = corpus
    self.compression = corpus.compression
    self.dir = corpus.full_text_path
    self.suffix = "." + self.format
    if self.compression:
      self.suffix += f".{compression}"
    super().__init__(corpus)

  def documents(self) -> Iterator[Document]:
    glob = f"**/*{self.suffix}"
    assert(self.dir.exists())
    for f in self.dir.glob(glob):
      yield Document(self.corpus, path = f)

  def __getitem__(self, key):
    return Document(self.corpus, path = self.dir / f"{key}{self.suffix}")

  def _iter_documents(self) -> Iterator[Tuple[str, str]]:
    for doc in self.documents():
      id = doc.id
      yield id, doc.full_text

class MetadataInput(TextInput):
  __doc__ = """
  For cases when the full text is stored inside the 
  metadata itself--as in common in, for example, social media datasets.

  Rather than parse multiple times, the best practice here is to convert to 
  feather *once* and then extract text and ids from the metadata column.
  """
  def __init__(self, corpus: "Corpus"):
    self.corpus = corpus
    self.text_field = corpus.text_field
    super().__init__(corpus)

  def __iter__(self) -> Iterator[Tuple[str, str]]:
    fin = self.corpus.metadata.nc_metadata_path
    table = feather.read_table(fin, columns = ['@id', self.text_field])
    table = ipc.open_file(self.corpus.root / "nonconsumptive_catalog.feather")
    for i in range(table.num_record_batches):
      batch = table.get_batch(i)
      for id, text in zip(batch['@id'], batch[self.text_field]):
        yield id.as_py(), text.as_py()