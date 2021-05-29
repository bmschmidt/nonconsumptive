from pathlib import Path
from .document import Document, tokenize, token_counts
import pyarrow as pa
from pyarrow import parquet, feather, RecordBatch 
from pyarrow import compute as pc
import json
import numpy as np
from bounter import bounter
from collections import defaultdict
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple
import polars as pl
from .data_storage import ArrowIDChunkedReservoir, ArrowReservoir, BatchIDIndex

from .prefs import prefs
from .metadata import Metadata
from .inputs import FolderInput, SingleFileInput, MetadataInput
from .transformations import transformations
import logging
logger = logging.getLogger("nonconsumptive")

class Corpus():

  def __init__(self, dir, texts = None, 
               metadata = None, cache_set = None, 
               text_options = {"format": None, "compression": None, "text_field": None},
               total_batches: Optional[int] = None,
               this_batch: int = 0):

    """Create a corpus

    Arguments:
    dir -- the location to save derived files.
    texts -- the location of texts to ingest (a file or a folder.) If None, no text can be read
             but metadata functions may still work.
    metadata -- the location of a metadata file with a suffix indicating 
                the file type (.parquet, .ndjson, .csv, etc.) If not passed, bare-bones metadata will 
                be created based on filenames.
    cache_set -- Which elements created by the nc_pipeline should be persisted to disk. If None,
                 defaults will be taken from .prefs().
    format    -- The text format used ('.txt', '.html', '.tei', '.md', etc.)
    compression -- Compression used on texts. Only '.gz' currently supported.
    total_batches -- how many chunks to process the corpus in.
    this_batch -- the batch number we're in right now. For multiprocessing, etc. zero-indexed.
    """
    if total_batches is None:
        self.uuid : str = "all"
    else:
      self.fraction = (this_batch, total_batches - 1)
      assert self.fraction[0] <= self.fraction[1]
      self.uuid : str = str(uuid.uuid1())
    self.fraction = 0
    if texts is not None: 
      self.full_text_path = Path(texts)
    else:
      self.full_text_path = None
    if metadata is not None:
      self.metadata_path = Path(metadata)
    else:
      self.metadata_path = None
    self.root: Path = Path(dir)
    self._metadata = None
    # which items to cache.
    self._cache_set = cache_set

    if "format" in text_options and text_options['format'] is not None:
      self.format = text_options["format"]
    else:
      self.format = "txt"

    if "compression" in text_options:
      self.compression = text_options["compression"]
    else:
      self.compression = None

    if "text_field" in text_options:
      self.text_field = text_options["text_field"]
    else:
      self.text_field = None
    self._texts = None
    self.transformations = {}

  def setup_input_method(self, full_text_path, method):
    if method is not None:
      return method
    if full_text_path.is_dir():
      return

  def batch_ids(self, uuid: str, id_type: str) -> Iterator[RecordBatch]:
    assert id_type in ["_ncid", "@id"]
    ids = BatchIDIndex(self, uuid)
    return ids.iter_ids(id_type)

  @property
  def texts(self):
    # Defaults based on passed input. This may become the only method--
    # it's not clear to me why one should have to pass both.
    if self._texts is not None:
      return self._texts
    if self.text_field is not None:
      if self.full_text_path is not None:
        raise ValueError("Can't pass both a full text path and a key to use for text inside the metadata.")
      self._texts = MetadataInput(self)
    elif self.full_text_path.is_dir():
      self.text_location = self.full_text_path
      assert self.text_location.exists()
      self._texts = FolderInput(self, compression = self.compression, format = self.format)
    elif self.full_text_path.exists():
      self._texts =  SingleFileInput(self, compression = self.compression, format = self.format)
    else:
      raise NotImplementedError("No way to handle desired texts. Please pass a file, folder, or metadata field.")
    return self._texts

  def clean(self, targets = ['metadata', 'tokenization', 'token_counts']):
    import shutil
    for d in targets:
      shutil.rmtree(self.root / d)

  @property
  def metadata(self) -> Metadata:
    if self._metadata is not None:
      return self._metadata
    self._metadata = Metadata(self, self.metadata_path)
    return self._metadata

  @property
  def cache_set(self):
    if self._cache_set:
      return self._cache_set
    else:
      return {}

  def path_to(self, id):
    p1 = self.full_text_path / (id + ".txt.gz")
    if p1.exists():
      return p1
    logger.error(FileNotFoundError("HMM"))
  
  @property
  def documents(self):
    if self.metadata and self.metadata.tb:        
      for id in self.metadata.tb["@id"]:
        yield Document(self, str(id))

  def get_document(self, id):
    return Document(self, id)

  @property
  def total_wordcounts(self, max_bytes:int=100_000_000) -> pa.Table:
    cache_loc = self.root / "wordids.parquet"
    if cache_loc.exists():
      logger.debug("Loading word counts from cache")
      return parquet.read_table(cache_loc)
    logger.info("Building word counts")
    counter = bounter(4096)
    for token, count in self.token_counts:
      dicto = dict(zip(token.to_pylist(), count.to_pylist()))
      counter.update(dicto)
    tokens, counts = zip(*counter.items())
    del counter
    tb = pa.table([pa.array(tokens, pa.utf8()), pa.array(counts, pa.uint32())],
      names = ['token', 'count']
    )
    tb = tb.take(pc.sort_indices(tb, sort_keys=[("count", "descending")]))
    tb = tb.append_column("wordid", pa.array(np.arange(len(tb)), pa.uint32()))
    if "wordids" in prefs('cache') or 'wordids' in self.cache_set:
      parquet.write_table(tb, cache_loc)
    return tb

  def __getattr__(self, key):
    try:
      return self.transformations[key]
    except KeyError:
      self.transformations[key] = transformations[key](self)
      return self.transformations[key]
      
  '''
  def feature_counts(self):
    if dir is None:
      dir = self.root / "feature_counts"
    for f in Path(dir).glob("*.parquet"):
      metadata = parquet.ParquetFile(f).schema_arrow.metadata.get(b'nc_metadata')
      metadata = json.loads(metadata.decode('utf-8'))
      yield (metadata, parquet.read_table(f))
  '''

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

  def write_feature_counts(self,
    single_files:bool = True,
    chunk_size:Optional[int] = None,
    batch_size:int = 250_000_000) -> None:
    
    """
    Write feature counts with metadata for all files in the document.
    dir: the location to place files into.
    chunking: whether to break up files into chunks of n words.
    single_files: Whether to write file per document, or group into batch
    batch_size bytes.
    """
    dir = self.root / Path(prefs("paths.feature_counts"))
    dir.mkdir(parents = True, exist_ok = True)
    bytes = 0
    counts = []
    batch_num = 0
    for doc in self.documents:
      try:
          wordcounts = doc.wordcounts
      except IndexError:
          print(doc.id)
          raise
      if single_files:
        parquet.write_table(pa.Table.from_batches([wordcounts]), (dir / f"{doc.id}.parquet").open("wb"))
      elif bytes > batch_size:
        counts.append(wordcounts)
        # Keep track of how big the table is.
        bytes += wordcounts.nbytes
        tab = pa.Table.from_batches(counts)
        feather.write_feather(tab, (dir / str(batch_num)).with_suffix("feather"))
        counts = []
        batch_num += 1
    if not single_files:
      # final flush
      tab = pa.Table.from_batches(counts)
      feather.write_feather(tab, (dir / str(batch_num)).with_suffix("feather"))
           
  def first(self) -> Document:
    for doc in self.documents:
      break
    return doc

  def random(self) -> Document:
    import random
    i = random.randint(0, len(self.metadata.tb) - 1)
    id = self.metadata.tb.column("@id")[i].as_py()
    return Document(self, id)
