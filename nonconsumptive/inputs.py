from pathlib import Path
from .document import Document
from typing import Callable, Iterator, Union, Optional, List, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
  from nonconsumptive import Corpus

import pyarrow as pa
from pyarrow import compute as pc
import logging
import gzip
from pyarrow import feather, ipc
from .data_storage import Node, BatchIDIndex
logger = logging.getLogger("nonconsumptive")

def input(path, corpus = None, format = "txt", compression=None):
  path = Path(path)
  if path.is_dir():
    return FolderInput(path, format, compression)  
  if path.full_text_path.exists():
    return SingleFileInput(path, format, compression)
  raise NotImplementedError("No such file")

class TextInput():
  """
  An object that defines a method for iterating across documents
  linked to IDs.

  There's one really important rule here:
  IT MUST BE IN A STABLE ORDER ACROSS MULTIPLE RUNS WITH THE SAME UUID!
  AND I DON'T KNOW HOW TO ENFORCE THIS!
  """
  def __init__(self, corpus):
    self.corpus = corpus

  def __getitem__(self, key):
    raise NotImplementedError("This text input type does not support random item access.")

  def iter_texts_for_ids(self, ids):
    for id in ids:
      id = id.as_py()
      yield self[id]

class SingleFileInput(TextInput):
  __doc__ = """

  One text per line (no returns in any document).
  id field separated from the rest of the doucment by a tab. (no tabs allowed in id.)

  """
  def __init__(self,
    dir:Path = Path("input.txt"),
    corpus: Optional["Corpus"] = None,
    compression: Optional[str] = None,
    format: str = "txt"):

    self.format = format
    self.compression = compression
    self.dir = dir
    super().__init__(corpus)

  def _offset_cache(self):
    pass

  def __getitem__(self, key):
    logger.warning("STOPGAP BEING USED FOR GETITEM")
    for id, text in self._iter_documents():
      if id == key:
        return text
    raise KeyError("Can't find")

  def _iter_documents(self) -> Iterator[Tuple[str, str]]:
    errored = []
    opener : function = open
    if self.compression == "gz":
      file = gzip.open(self.dir, 'rt')
    else:
      file = open(self.dir)
    for line in file:
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
      dir: Path,
      corpus: "Corpus" = None,
      format:str = "txt",
      compression: Optional[str] = None):
    assert isinstance(dir, Path)
    self.format = format
    self.corpus = corpus
    self.compression = compression
    self.dir = dir
    self.suffix = "." + self.format
    if self.compression:
      self.suffix += f".{compression}"
    super().__init__(corpus)

  def __iter__(self):
    for id in self.ids():
      yield self[id]

  def __getitem__(self, key):
    path = self.dir / f"{key}{self.suffix}"
    if self.compression == "gz":
      return gzip.open(path, 'rt').read()
    elif self.compression is None:
      return path.open().read()
    raise NotImplementedError("No method for " + self.compression)

  def ids(self):
    for file in walk_path(self.dir):
      name = file.name
      if self.suffix in name:
        yield name.replace(self.suffix, "")

def walk_path(path):
  # Sort to preserve consistent order
  assert isinstance(path, Path)
  children = [*Path(path).iterdir()]
  children.sort()
  for p in children:
    if p.is_dir(): 
        yield from walk_path(p)
        continue
    yield p.resolve()

class PandocInput(TextInput):
  def __init__(self,       
      dir: Path,
      corpus: "Corpus" = None,
      format:str = "md",
      compression: Optional[str] = None):
      super().__init__(corpus)

  def __iter__(self):
    pass

class MetadataInput(TextInput):
  __doc__ = """
  For cases when the full text is stored inside the 
  metadata itself--as in common in, for example, social media datasets.

  Rather than expensively parse multiple times, the best practice here is to convert to 
  feather *once* and then extract text and ids from the metadata column.
  """
  def __init__(self, path, text_field = "@text", corpus = None, **kwargs):
    self.text_field = text_field
    self.path = path
    self._tb = None
    super().__init__(corpus)

  def iter_texts_for_ids(self, ids) -> Iterator[str]:
    indices = pc.index_in(ids, value_set = self.tb["@id"])
    for index in indices:
      yield self.tb[self.text_field][index.as_py()].as_py()

  @property
  def tb(self):
    if self._tb is not None:
      return self._tb
    self._tb = feather.read_table(self.path, columns = [self.text_field, "@id"])
    return self._tb
    
  def ids(self) -> Iterator[str]:
    if self.corpus.uuid is None:
      for id in self._tb["@id"]:
        yield id.as_py()
    else:
      for id in feather.read_table(self.corpus.root / f"bookstacks/{self.corpus.uuid}.feather")['@id']:
        yield id

  def __iter__(self) -> Iterator[str]:
    for text in feather.read_table(self.path, columns = [self.text_field])[self.text_field]:
      yield text.as_py()
    