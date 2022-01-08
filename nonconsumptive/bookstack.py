from __future__ import annotations

from pathlib import Path
import pyarrow as pa
from pyarrow import parquet, feather, RecordBatch 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
  
from collections import defaultdict
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple, Set, Dict
import polars as pl
from .data_storage import BatchIDIndex
from .transformations import transformations
import logging


logger = logging.getLogger("nonconsumptive")

class Bookstack():

  """
  A bookstack represents a self-contained slice of corpus, including 
  metadata, tokenizations, and other processes.

  
  """

  TARGET_BATCH_SIZE = 1024 * 1024 * 16 

  def __init__(self, parent_corpus: nc.Corpus, uuid):
    """
    Define a single stack in a corpus.

    uuid: a unique identifier for this stack.
    """
    self.is_bookstack = True
    self.uuid = uuid
    self._ids = None
    self.transformations : dict = {}
    self.corpus = parent_corpus
    self.root = (parent_corpus).root
    self._metadata = None

  @property
  def ids(self):
    if self._ids is not None:
      return self._ids

    self._ids = feather.read_table(self.corpus.root / f"metadata/{self.uuid}.feather", 
      columns = ["@id", "nc:id"])
    return self._ids

  @property
  def total_wordcounts(self):
    return self.corpus.total_wordcounts

  @property
  def metadata(self):
    fin = (self.corpus.root / "metadata" / self.uuid).with_suffix(".feather")
    if self._metadata is not None:
      return self._metadata
    self._metadata = feather.read_table(fin)
    return self._metadata
  @property
  def bookstacks(self):
    raise RecursionError("A stack can't contain more stacks")

  def get_transform(self, key):
    """
    Transformations are applied against a bookstack.
    """
    if not key in self.transformations:
      self.transformations[key] = transformations[key](self)
    return self.transformations[key]

def running_processes(workerlist):
    running = False
    for worker in workerlist:
        if worker.is_alive():
            running = True
    return running

from pathlib import Path
from pyarrow import parquet
import pyarrow as pa
from typing import Union

class Bookstacks:
    """
    A set of bookstacks structured as a parquet dataset.
    Should be called "Library", but all the computer 
    people messed that up with a different set of associations.
    """
    def __init__(self, loc: Union[Path, str], mode = 'r'):
        self.dir = Path(loc)
        assert mode in 'wr'
        if mode == 'r':
          assert self.dir.is_dir()
        
    @property
    def schema(self):
        f0, *_ = self.files()
        return parquet.ParquetFile(f0).schema_arrow
    
    def meta(self):
        return [p.name for p in self.schema]
    
    def files(self):
      # List all the files in this bookstack in lexicographically sorted order.
      fs = [*self.dir.glob("**/*.parquet")]
      fs.sort()
      return fs
            
    def __repr__(self):
        return f"A bookstack set at {self.dir} with {len([*self.files()])} stacks."
    
    def metadata(self):
        names = [f.name for f in self.schema if not f.name.startswith("nc:")]
        batches = []
        for file in self.files():
            print(file, end="\r")
            batches.extend(parquet.read_table(file, columns=names).to_batches())
        return pa.Table.from_batches(batches)