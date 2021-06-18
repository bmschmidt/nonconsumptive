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
from bounter import bounter
from collections import defaultdict
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple, Set, Dict
import polars as pl
from .data_storage import BatchIDIndex
import uuid
from .metadata import Metadata, cat_from_filenames
from .inputs import TextInput, FolderInput, SingleFileInput, MetadataInput
from .transformations import transformations
import logging

from multiprocessing import Process, Queue
from queue import Empty
import time

logger = logging.getLogger("nonconsumptive")

class Bookstack():
  def __init__(self, parent_corpus: nc.Corpus, uuid):
    self.is_bookstack = True
    self.uuid = uuid
    self.ids = feather.read_table(parent_corpus.root / f"bookstacks/{uuid}.feather", columns = ["@id", "_ncid"])
    self.transformations : dict = {}
    self.corpus = parent_corpus
    self.root = (parent_corpus).root
    self._cache_set = (parent_corpus).cache_set
    self.metadata = (parent_corpus).metadata
    self.cache_set = parent_corpus.cache_set

  @property
  def total_wordcounts(self):
    return self.corpus.total_wordcounts

  def create_bookstack_plan(self):
    raise NotImplementedError()

  @property
  def bookstacks(self):
    raise RecursionError("A stack can't contain more stacks")

  def get_transform(self, key):
    if not key in self.transformations:
      self.transformations[key] = transformations[key](self)
    return self.transformations[key]

def running_processes(workerlist):
    running = False
    for worker in workerlist:
        if worker.is_alive():
            running = True
    return running