from __future__ import annotations

from pathlib import Path
from .document import Document
import pyarrow as pa
from pyarrow import parquet, feather, RecordBatch, ipc
from pyarrow import compute as pc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
import numpy as np
from bounter import bounter
from collections import defaultdict
from typing import Any, DefaultDict, Iterator, Union, Optional, List, Tuple, Set, Dict, Iterable
import polars as pl
from .data_storage import BatchIDIndex
import uuid
from .metadata import Metadata, cat_from_filenames
from .inputs import TextInput, FolderInput, SingleFileInput, MetadataInput
from .transformations import transformations
from .bookstack import Bookstack
import logging

from multiprocessing import Process, Queue, Pool
from queue import Empty
import time

logger = logging.getLogger("nonconsumptive")

StrPath = Union[str, Path]


class Corpus():
  def __init__(self, dir: StrPath, texts: Optional[StrPath] = None,
               metadata: Optional[StrPath] = None,
               cache_set: Set[str] = {"tokenization", "word_counts", "document_lengths"}, 
               text_options: Dict[str, Optional[str]] = {},
               metadata_options: Dict[str, Optional[str]] = {"id_field": None},
               batching_options : Dict[str, int]= {"batch_size": 2**16},
               bookstacks : Optional[StrPath] = None, 
               only_stacks : Optional[List[str]] = None):

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

    # Allow cloning.
    self.kwargs = {
      "dir": dir,
      "texts": texts,
      "metadata": metadata,
      "cache_set": cache_set,
      "text_options": text_options,
      "metadata_options": metadata_options,
      "batching_options": batching_options,
      "bookstacks": bookstacks,
      "only_stacks": only_stacks
    }

    self.uuid : str = None
#    self.metadata_options = metadata_options
    self.root: Path = Path(dir)
    self.root.mkdir(exist_ok = True)
    self.only_stacks = only_stacks
    self.batch_size = batching_options["batch_size"]
    self.text_options = text_options
    # A place where children can stash things they need to create 
    # themselves. Use with care, because not thread-safe.
    self.slots : Dict[str, Any]= {}
    if bookstacks:
      self.input_bookstacks : Union[Path, None] = Path(bookstacks)
    else:
      self.input_bookstacks = None

    if metadata is None and bookstacks is None:
      self.setup_texts(texts, **text_options)
      if not (self.root / "metadata").exists():
        dummy_metadata = self.root / 'tmp_metadata.feather'
        ids = pa.table([pa.array(self.text_input.ids(), pa.string())], names = ["@id"])
        feather.write_feather(ids, dummy_metadata)
        self.setup_metadata(dummy_metadata, **metadata_options)
        dummy_metadata.unlink()
      else:
        self.setup_metadata(self.root / "metadata")
    elif metadata is not None and bookstacks is None:
      self.setup_metadata(metadata, **metadata_options)
      if texts is None:
        texts = Path(self.metadata.path)
        try:
          assert "metadata_field" in text_options or self.input_bookstacks
        except AssertionError:
          raise KeyError("Must pass `metadata_field` variable for catalog "
                        "if no text location passed")
        self.setup_texts(texts, from_metadata = True, **text_options)
      else:
        self.setup_texts(texts, from_metadata = False, **text_options)

    elif metadata is None and bookstacks is not None:
      self.setup_metadata(self.input_bookstacks, **metadata_options)

    # which items to cache.
    self._cache_set = cache_set

    # Later, create the bookstacks where transformations will be stored.
    self._total_wordcounts = None
    self._stacks = None

  def batch_ids(self, uuid: str, id_type: str) -> Iterator[RecordBatch]:
    assert id_type in ["nc:id", "@id"]
    ids = BatchIDIndex(self)
    return ids.iter_ids(id_type)

  @property
  def text_input(self):
    return self._texts

  def setup_metadata(self, location, **kwargs):
    location = Path(location)
    self._metadata = Metadata(self, location, **kwargs)

  def setup_texts(self, texts: StrPath, from_metadata = False,
                  metadata_field : Optional[str] = None, **kwargs):
    self._texts : Optional[TextInput] = None
    if texts is None:
      return
    texts = Path(texts)
    assert texts.exists()
    self.text_location = texts

    if from_metadata:
      self._texts = MetadataInput(metadata_field = metadata_field, corpus = self, **kwargs)
      return
    if texts.is_dir():
      self._texts = FolderInput(texts, **kwargs)
    else:
      self._texts =  SingleFileInput(texts, corpus = self, **kwargs)

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
      self._total_wordcounts = feather.read_table(cache_loc)[:1_000_000]
      return self._total_wordcounts

    logger.info("Building word counts")

    MAX_MEGABYTES = 4096
    import bounter
    counter = bounter.bounter(MAX_MEGABYTES)
    stack_size = 0
    stack = []
    # Do the first pass of counting precisely using polars.
    # Avoids a bunch of unnecessary typescasts and 
    # unparallelized additions.
    for i, bstack in enumerate(self.bookstacks):
      transform = bstack.get_transform("unigrams")
      for wordcounts in transform:
          # First merge the documents; then split words from counts
          tokens, counts = wordcounts[0].flatten()
          batch = pa.record_batch([tokens.flatten(), counts.flatten()], ['token', 'count'])
          stack.append(batch)
          stack_size += batch.nbytes
          # Use one-tenth the stack size to store here.
          if stack_size >= (MAX_MEGABYTES * 1024 * 1024 / 10) or i == (len(self.bookstacks) - 1):
            logging.debug("Writing to bounter")
            stuck : pl.DataFrame = pl.from_arrow(pa.Table.from_batches(stack))
            stack = []
            count = stuck.groupby("token")['count'].sum()
            del stuck
            stack_size = 0
            #logger.info(f"Flushing counts at bookstack {i} size {stack_size / 1024 / 1024:.02f}MB")
            counter.update(
              dict(zip(count['token'].to_list(), count['count_sum'].to_list()))
            )
            del count

    tokens, counts = zip(*counter.items())
    del counter
    tb: pa.Table = pa.table([pa.array(tokens, pa.utf8()), pa.array(counts, pa.uint32())],
      names = ['token', 'count']
    )
    del tokens
    del counts
    tb = tb.take(pc.sort_indices(tb, sort_keys=[("count", "descending")]))
    tb = tb.append_column("wordid", pa.array(np.arange(len(tb)), pa.uint32()))
    if True or "wordids" in self.cache_set:
      # Materialize to save memory.
      feather.write_feather(tb, cache_loc)
      self._total_wordcounts = feather.read_table(cache_loc)[:1_000_000]
      return self._total_wordcounts
    else:
      self._total_wordcounts = tb
      return tb

  def unigrams(self):
    yield from self.iter_over("unigrams")
  
  def encoded_wordcounts(self):
    # Enforce complete build of the wordcounts first.
    _ = self.total_wordcounts
    yield from self.iter_over("encoded_wordcounts")

  def cache(self, transformation):
    """
    Save the representations of 'transformation' to disk at the corpus location.
    """
    self.cache_set.add(transformation)
    for _ in self.iter_over(transformation):
      pass

  def table(self, key):
    tabs = [stack.get_transform(key).table for stack in self.bookstacks]
    return pa.concat_tables(tabs)

  def text(self):
    yield from self.iter_over('text')

  def tokenization(self):
    yield from self.iter_over('tokenization')

  def bigrams(self):
    yield from self.iter_over('bigrams')

  def SRPFeatures(self):
    yield from self.iter_over('SRPFeatures')

  def document_lengths(self):
    yield from self.iter_over('document_lengths')

  def iter_over(self, key, ids = None):
    threads = 1

    bookstack_queue = Queue(threads)
    results_queue = Queue(threads * 2)
    assert len(self.bookstacks) > 0, "No bookstacks to iterate over"
    for stack in self.bookstacks:
      transformation = stack.get_transform(key)
      if ids is None:
        yield from transformation
      elif ids in {"@id", "nc:id"}:
        yield from transformation.iter_with_ids(ids)
      else:
        raise ValueError(f'ids must be in {"@id", "nc:id"}')
      
  def to_parquet(self, transformations = ["unigrams", "document_length", "SRP"]):
    # Writes a parquet file including derived metadata.
    schema = self.metadata.load_processed_catalog(columns = None)
    
    parquet.ParquetFile(self.root / "export.parquet", )  
    

  @property
  def bookstacks(self):
    """
    A set of reasonably-sized chunks of text to work with.
    """
    if self._stacks is not None:
      return self._stacks

    self._stacks = []
    for f in self.metadata.path.glob("*.feather"):
      name = f.with_suffix("").name
      if self.only_stacks and name not in self.only_stacks:
        continue
      self._stacks.append(Bookstack(self, name))

    return self._stacks

  def audit(self, field):
    for stack in self.bookstacks:
      pass

  """
  @property
  def wordids(self):
    ""
    Returns wordids in a dict format. 
    ""
    logger.warning("Using dict method for wordids, which is slower.")
    w = self.total_wordcounts
    tokens = w['token']
    counts = w['count']
    return dict(zip(tokens.to_pylist(), counts.to_pylist()))
  """

  def first(self) -> Document:
    for doc in self.documents:
      break
    return doc

  def random(self) -> Document:
    import random
    i = random.randint(0, len(self.metadata.tb) - 1)
    id = self.metadata.tb.column("@id")[i].as_py()
    return Document(self, id)

  """
  def _load_bookstack_plan(self, outdir):
    ""
    Load a passed bookstack.
    ""
    if self.input_bookstacks is None:
      return
    i = 0
    for f in Path(self.input_bookstacks).glob("*.parquet"):
      id_slice = parquet.read_table(f, columns = [self.metadata.id_field])[self.metadata.id_field]
      top = i + len(id_slice)
      tb = pa.table([
        id_slice,
        pa.array(range(i, top), pa.uint32())], 
        names = ["@id", "nc:id"])
      outpath = outdir / (f.with_suffix(".feather").name)
      feather.write_feather(tb, outpath, chunksize = 10_000_000)
      i += len(id_slice)
      yield f.with_suffix("").name
  """

  def multiprocess(self, task, processes = 6, stacks_per_process = 3):
    ids = [[]]
#    ids = [p.id for p in self.bookstacks]
    for stack in self.bookstacks:
      if len(ids[-1]) < stacks_per_process:
        ids[-1].append(stack.uuid)
      else:
        ids.append([stack.uuid])
    with Pool(processes) as p:
      p.starmap(subprocess, [
        (task, ids, self.kwargs) for ids in ids])

def subprocess(task, batches, kwargs):
  kwargs['only_stacks'] = batches
  corp = Corpus(**kwargs)
  corp.cache(task)
  