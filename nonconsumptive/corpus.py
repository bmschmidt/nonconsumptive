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
from .bookstack import Bookstack
import logging

from multiprocessing import Process, Queue
from queue import Empty
import time

logger = logging.getLogger("nonconsumptive")

StrPath = Union[str, Path]

class Corpus():
  def __init__(self, dir: StrPath, texts: Optional[StrPath] = None, 
               metadata: Optional[StrPath] = None,
               cache_set: Optional[Set[str]] = None, 
               text_options: Dict[str, Optional[str]] = {},
               metadata_options: Dict[str, Optional[str]] = {"id_field": None},
               batching_options = {"batch_size": 2**16}):

    """Create a corpus

    Arguments:
    dir -- the location to save derived files.
    texts -- the location of texts to ingest (a file or a folder.) If None, no text can be read
             but metadata functions may still work.
    metadata -- the location of a metadata file with a suffix indicating 
                the file type (.parquet, .ndjson, .csv, etc.) If not passed, bare-bones metadata will 
                be created based on filenames.
    cache_set -- Which elements created by the nc_pipeline should be persisted to disk.
    format    -- The text format used ('txt', 'html', 'tei', 'md', etc.)
    compression -- Compression used on texts. Only '.gz' currently supported.
    total_batches -- how many chunks to process the corpus in.
    this_batch -- the batch number we're in right now. For multiprocessing, etc. zero-indexed.
    """

    self.uuid : str = None
    self.metadata_options = metadata_options
    self.root: Path = Path(dir)
    self.root.mkdir(exist_ok = True)

    if metadata is None:
      self.setup_texts(texts, **text_options)
      dummy_metadata = self.root / 'tmp_metadata.feather'
      ids = pa.table([pa.array(self.text_input.ids(), pa.string())], names = ["@id"])
      feather.write_feather(ids, dummy_metadata)
      self.setup_metadata(dummy_metadata, **metadata_options)
      dummy_metadata.unlink()
    else:
      self.setup_metadata(metadata, **metadata_options)
      if texts is None:
        texts = Path(self.metadata.path)
        try:
          assert "text_field" in text_options
        except AssertionError:
          raise KeyError("Must pass `text_field` variable for catalog "
                        "if no text location passed")
        self.setup_texts(texts, from_metadata = True, **text_options)
      else:
        self.setup_texts(texts, from_metadata = False, **text_options)

    # which items to cache.
    self._cache_set = cache_set

    # Later, create the bookstacks where transformations will be stored.
    self._total_wordcounts = None
    self._stacks = None

  def batch_ids(self, uuid: str, id_type: str) -> Iterator[RecordBatch]:
    assert id_type in ["_ncid", "@id"]
    ids = BatchIDIndex(self)
    return ids.iter_ids(id_type)

  @property
  def text_input(self):
    return self._texts

  def setup_metadata(self, location, **kwargs):
    location = Path(location)
    self._metadata = Metadata(self, location)

  def setup_texts(self, texts: StrPath, from_metadata = False, text_field : Optional[str] = None, **kwargs):
    # Defaults based on passed input. This may become the only method--
    # it's not clear to me why one should have to pass both.
    texts = Path(texts)
    assert texts.exists()
    self._texts : TextInput

    if from_metadata:
      self._texts = MetadataInput(texts, text_field = text_field, **kwargs)
      return
    if texts.is_dir():
      self._texts = FolderInput(texts, **kwargs)
    else:
      self._texts =  SingleFileInput(texts, **kwargs)

  def clean(self, targets: List[str] = ['metadata', 'tokenization', 'token_counts']):
    import shutil
    for d in targets:
      shutil.rmtree(self.root / d)

  @property
  def metadata(self) -> Metadata:
    return self._metadata

  @property
  def cache_set(self) -> Set[str]:
    if self._cache_set:
      return self._cache_set
    else:
      return set([])
  
  @property
  def documents(self) -> Iterator[nc.Document]:
    if self.metadata and self.metadata.tb:        
      for id in self.metadata.tb["@id"]:
        yield Document(self, str(id))

  def get_document(self, id):
    return Document(self, id)

  @property
  def total_wordcounts(self) -> pa.Table:
    if self._total_wordcounts is not None:
      return self._total_wordcounts
    cache_loc = self.root / "wordids.feather"
    if cache_loc.exists():
      logger.debug("Loading word counts from cache")
      self._total_wordcounts = feather.read_table(cache_loc)
      return self._total_wordcounts
    logger.info("Building word counts")
    counter = bounter(4096)
    for token, count in self.iter_over("token_counts"):
      dicto = dict(zip(token.to_pylist(), count.to_numpy()))
      counter.update(dicto)
    tokens, counts = zip(*counter.items())
    del counter
    tb: pa.Table = pa.table([pa.array(tokens, pa.utf8()), pa.array(counts, pa.uint32())],
      names = ['token', 'count']
    )
    del tokens
    del counts
    tb = tb.take(pc.sort_indices(tb, sort_keys=[("count", "descending")]))
    tb = tb.append_column("wordid", pa.array(np.arange(len(tb)), pa.uint32()))
    if "wordids" in self.cache_set:
      # Materialize to save memory.
      feather.write_feather(tb, cache_loc)
      self._total_wordcounts = feather.read_table(cache_loc)
      return self._total_wordcounts
    else:
      self._total_wordcounts = tb
      return tb

  def token_counts(self):
    yield from self.iter_over("token_counts")
  
  def encoded_wordcounts(self):
    # Enforce build of this *first*.
    wordcounts = self.total_wordcounts
    yield from self.iter_over("encoded_wordcounts")

  def tokenization(self):
    yield from self.iter_over('tokenization')

  def bigrams(self):
    yield from self.iter_over('bigrams')

  def SRPFeatures(self):
    yield from self.iter_over('SRPFeatures')

  def document_lengths(self):
    yield from self.iter_over('document_lengths')

  def iter_over(self, key):
    threads = 1

    bookstack_queue = Queue(threads)
    results_queue = Queue(threads * 2)
    for stack in self.bookstacks:
      yield from stack.get_transform(key)

    """
    def feed_stacks():
      for stack in self.bookstacks:
        bookstack_queue.put(stack)

    def feed_results():
      while True:
        try:
          stack = bookstack_queue.get(timeout = 1)
        except Empty:
          return
        for item in stack.get_transform(key):
           results_queue.put(item)

    feeder = Process(target = feed_stacks).start()
    workers = []

    for i in range(8):
      t = Process(target = feed_results)
      t.start()
      workers.append(t)
    
    while True:
      try:
        yield results_queue.get_nowait()
      except Empty:
        if running_processes(workers):
          time.sleep(1/100)
        else:
          break
      """
      
  @property
  def bookstacks(self):
    """
    A set of reasonably-sized chunks of text to work with.
    """
    if self._stacks is not None:
      return self._stacks
    self._stacks = []
    ids = self._create_bookstack_plan(2 ** 8)
    for id in ids:
      self._stacks.append(Bookstack(self, id))
    return self._stacks

  @property
  def wordids(self):
    """
    Returns wordids in a dict format. 
    """
    logger.warning("Using dict method for wordids, which is slower.")
    w = self.total_wordcounts
    tokens = w['token']
    counts = w['count']
    return dict(zip(tokens.to_pylist(), counts.to_pylist()))
           
  def first(self) -> Document:
    for doc in self.documents:
      break
    return doc

  def random(self) -> Document:
    import random
    i = random.randint(0, len(self.metadata.tb) - 1)
    id = self.metadata.tb.column("@id")[i].as_py()
    return Document(self, id)

  def _create_bookstack_plan(self, size = None):
    if size is None:
      size = 2 ** 16
    dir = self.root / "bookstacks"
    dir.mkdir(exist_ok = True)
    cat = feather.read_table(self.metadata.path, columns = ["@id"])
    ids = cat['@id']
    batch_num = 0
    i = 0
    stack_names = []
    while i < len(cat):
      top = min(i + size, len(cat))
      slice = ids[i:top]
      tb = pa.table([
        slice,
        pa.array(range(i, top), pa.uint32())], 
        names = ["@id", "_ncid"])
      name = f"{batch_num:05d}"
      feather.write_feather(tb, dir / f"{name}.feather", chunksize=size + 1)
      stack_names.append(name)
      i += size
      batch_num += 1
    return stack_names
