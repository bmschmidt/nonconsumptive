from __future__ import annotations

from pathlib import Path
from .document import Document
import pyarrow as pa
from pyarrow import parquet, feather, RecordBatch 
from pyarrow import compute as pc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
import numpy as np
from collections import defaultdict
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple, Set, Dict
import polars as pl
from .data_storage import BatchIDIndex
import uuid
from .inputs import TextInput, FolderInput, SingleFileInput, MetadataInput
from .transformations import transformations
import logging

from multiprocessing import Process, Queue
from queue import Empty
import time

logger = logging.getLogger("nonconsumptive")

class Bookstack():

  TARGET_BATCH_SIZE = 1024 * 1024 * 16 

  def __init__(self, parent_corpus: nc.Corpus, uuid):
    self.is_bookstack = True
    self.uuid = uuid
    self._ids = None
    self.transformations : dict = {}
    self.corpus = parent_corpus
    self.root = (parent_corpus).root

  @property
  def ids(self):
    if self._ids is not None:
      return self._ids

    self._ids = feather.read_table(self.corpus.root / f"bookstacks/{self.uuid}.feather", 
      columns = ["@id", "_ncid"])
    return self._ids

  @property
  def total_wordcounts(self):
    return self.corpus.total_wordcounts

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

class Athenaeum:
    """
    A set of bookstacks structured as a parquet dataset.
    Should be called "Library", but for some reason all the computer 
    people messed that up with a different set of associations.
    """
    def __init__(self, loc: Union[Path, str]):
        self.dir = Path(loc)
        assert self.dir.is_dir()
        
    @property
    def schema(self):
        f0 = self.files().__next__()
        return parquet.ParquetFile(f0).schema_arrow
    
    def meta(self):
        return [p.name for p in self.schema]
    
    def files(self):
        for f in self.dir.glob("**/*.parquet"):
            yield f
            
    def __repr__(self):
        return f"A bookstack at {self.dir} with {len([*self.files()])} stacks."
    
    def metadata(self):
        names = [f.name for f in self.schema if not f.name.startswith("nc:")]
        batches = []
        for file in self.files():
            print(file, end="\r")
            batches.extend(parquet.read_table(file, columns=names).to_batches())
        return pa.Table.from_batches(batches)