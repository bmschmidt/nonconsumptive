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