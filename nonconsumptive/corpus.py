from pathlib import Path
from .document import Document, tokenize, token_counts
import pyarrow as pa
from pyarrow import parquet, feather
from pyarrow import compute as pc
import json
import numpy as np
from bounter import bounter
from collections import defaultdict
from typing import DefaultDict, Iterator, Union, Optional, List
import polars as pl

from .prefs import prefs
from .metadata import Metadata
from .data_storage import ArrowIDBatchReservoir
from .inputs import FolderInput, SingleFileInput, MetadataInput
from .arrow_helpers import batch_id
_Path = Union[str, Path]

import logging
logger = logging.getLogger("nonconsumptive")

class Corpus():

  def __init__(self, dir, texts = None, 
               metadata = None, cache_set = None, 
               text_options = {"format": None, "compression": None, "text_field": None}):

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

    """
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

  def setup_input_method(self, full_text_path, method):
    if method is not None:
      return method
    if full_text_path.is_dir():
      return

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

  @property 
  def bigrams(self):
    """
    returns a reservoir that iterates over tokenized 
    arrow batches of documents as arrow arrays. 
    """
    tokenizer = Bigrams(self)
    return tokenizer

  @property 
  def trigrams(self):
    """
    returns a reservoir that iterates over tokenized 
    arrow batches of documents as arrow arrays. 
    """
    tokenizer = Trigrams(self)
    return tokenizer

  @property 
  def quadgrams(self):
    """
    returns a reservoir that iterates over tokenized 
    arrow batches of documents as arrow arrays. 
    """
    tokenizer = Quadgrams(self)
    return tokenizer

  @property 
  def quintgrams(self):
    """
    returns a reservoir that iterates over tokenized 
    arrow batches of documents as arrow arrays. 
    """
    tokenizer = Quintgrams(self)
    return tokenizer

  @property 
  def tokenization(self):
    """
    returns a reservoir that iterates over tokenized 
    arrow batches of documents as arrow arrays. 
    """
    tokenizer = Tokenization(self)
    return tokenizer

  @property
  def token_counts(self):
    """
    Returns a reservoir that iterates over grouped
    counts of tokens.
    """
    token_counts = TokenCounts(self)
    return token_counts

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
    counter = bounter(2048)
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

  def feature_counts(self, dir = None):
    if dir is None:
      dir = self.root / "feature_counts"
    for f in Path(dir).glob("*.parquet"):
      metadata = parquet.ParquetFile(f).schema_arrow.metadata.get(b'nc_metadata')
      metadata = json.loads(metadata.decode('utf-8'))
      yield (metadata, parquet.read_table(f))

  @property
  def wordids(self):
    """
    Returns wordids in a dict format. 
    """
    w = self.total_wordcounts
    tokens = w['token']
    counts = w['count']
    return dict(zip(tokens.to_pylist(), counts.to_pylist()))

  @property
  def encoded_wordcounts(self):
    i = 0
    bookid_lookup = self.metadata.id_to_int_lookup
    wordids = pl.from_arrow(self.total_wordcounts.select(["token", "wordid"]))
    batches = []
    cache_size = 0

    for batch in self.token_counts:
        try:
            id = batch_id(batch)
            bookid = bookid_lookup[id]
        except KeyError:
            raise
        tab = pa.table({
            'bookid': pa.array(np.full(len(batch), bookid, dtype = np.uint32), pa.uint32()),
            'token': batch['token'],
            'count': batch['count']
        })
        batches.append(tab)
        cache_size += tab.nbytes
        if cache_size > 5_000_000:
          remainder = pl.from_arrow(pa.concat_tables(batches))
          q = wordids.join(remainder, left_on=["token"], right_on=["token"], how="inner", )
          for b in q.to_arrow().select(["bookid", "wordid", "count"]).to_batches():
              yield b          
          cache_size = 0
          batches = []
    remainder = pl.from_arrow(pa.concat_tables(batches))
    q = wordids.join(remainder, left_on=["token"], right_on=["token"], how="inner", )
    for b in q.to_arrow().select(["bookid", "wordid", "count"]).to_batches():
        yield b
      
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

class Tokenization(ArrowIDBatchReservoir):
  name = "tokenization"
  arrow_schema = pa.schema({
    'token': pa.utf8()
  })

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    for id, text in self.corpus.texts:
      tokens = tokenize(text)
      yield pa.record_batch(
        [tokens],
        schema=self.arrow_schema.with_metadata({"@id": id})
      )

class TokenCounts(ArrowIDBatchReservoir):
  name = "token_counts"
  arrow_schema = pa.schema({
    'token': pa.utf8(),
    'count': pa.uint32()
  })

  def __init__(self, *args, **kwargs):
    self.ids_two = []
    self.doc_lengths = []
    super().__init__(*args, **kwargs)

  def flush_wordcounts(self, uuid):
    parent = self.corpus.root / "document_lengths"
    parent.mkdir(exist_ok = True)
    path = (parent / uuid).with_suffix(".feather")
    tab = pa.table(
      {"id": pa.array(self.ids_two, pa.string()),
      "count": pa.array(self.doc_lengths, pa.uint32())}
    )
    feather.write_feather(tab, path)
    self.ids_two = []
    self.doc_lengths = []

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    current_uuid = None
    for batch in self.corpus.tokenization:
      id = batch_id(batch)
      counts = token_counts(batch['token'], id)
      if self.current_file_uuid != current_uuid and current_uuid is not None:
        self.flush_wordcounts(current_uuid)
        current_uuid = self.current_file_uuid
      total_wordcount = pc.sum(counts['count']).as_py()
      self.ids_two.append(id)
      self.doc_lengths.append(total_wordcount)
      yield counts

    self.flush_wordcounts(self.current_file_uuid)

class Ngrams(ArrowIDBatchReservoir):
  def __init__(self, ngrams: int, corpus: Corpus, end_chars: List[str] = [], beginning_chars: List[str] = [], **kwargs):
    """

    Creates an (optionally cached) iterator over record batches of ngram counts.
    Each batch is a single document, with columns ['token1', 'token2', ... , 'token{n}', count]

    ngrams: an integer. Size of the ngrams to construct.
    end_chars: a list of regular expression (re2 compatible) to treat as the *end* of an n-gram.
               For instance, [r"[\.\?\!]"] would attach sentence-ending punctuators to the end of
               n-grams but not the beginning, and not allow ngrams from separate sentence.
    beginning_chars: a list of regular expression (re2 compatible) to treat as the *beginning* of an n-gram.
               For instance, ["“", "<"] would attach opening curly quotes and angle brackets to n-grams with the letters that
               follow. (This is provided experimentally, may be removed because it doesn't seem very useful.
               But perhaps there are languages where it helps.)

    *args, **kwargs: passed to ArrowReservoir.

    """
    self.ngrams = ngrams
    self.name = f"{ngrams}gram_counts"
    self.end_chars = end_chars
    self.beginning_chars = beginning_chars
    super().__init__(corpus, **kwargs)
#    for n in range(ngrams):
#      self.arrow_schema[f"token{n}"] = pa.utf8()
#    self.arrow_schema['count'] = pa.uint32()

  def _from_local(self):
    pass

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    ngrams = self.ngrams
    for batch in self.corpus.tokenization:
      id = batch_id(batch)
      zengrams = ngrams - 1 # Zero-indexed ngrams--slightly more convenient here.
      tokens = batch['token']
      cols = {}
      for n in range(ngrams):
          if n == 0:
              cols["token1"] = tokens[:-zengrams]
          elif n == ngrams - 1:
              cols[f"token{n+1}"] = tokens[n:]
          else:
              cols[f"token{n+1}"] = tokens[n:-(zengrams-n)]
      ngram_tab = pa.table(cols)
      if len(ngram_tab) < ngrams:
        empty = [[]]
        # Still register that we saw the id even if it's not long enough for ngrams
        yield pa.RecordBatch.from_arrays(empty * len(self.arrow_schema.names), schema=self.arrow_schema.with_metadata({'@id': batch_id(batch)}))
        continue
      # Temporarily pass through pandas because https://github.com/pola-rs/polars/issues/668
      t = pl.from_pandas(ngram_tab.to_pandas())
      counts = t.groupby([f"token{i+1}" for i in range(ngrams)])\
          .agg([pl.count("token1").alias("count")])

      cols = []
      for i in range(ngrams):
          cols.append(counts[f'token{i+1}'].to_arrow().cast(pa.string()))
      cols.append(counts['count'].to_arrow())
      yield pa.RecordBatch.from_arrays(cols, schema=self.arrow_schema.with_metadata({'@id': batch_id(batch)}))

class Bigrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  name = "bigrams"
  arrow_schema = pa.schema(
    {
      'token1': pa.string(),
      'token2': pa.string(),
      'count': pa.uint32()
    }
  )
  def __init__(self, corpus, **kwargs):
    super().__init__(ngrams=2, corpus=corpus, **kwargs)

class Trigrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  arrow_schema = pa.schema(
    {
      'token1': pa.string(),
      'token2': pa.string(),
      'token3': pa.string(),
      'count': pa.uint32()
    }
  )
  def __init__(self, corpus, **kwargs):
    super().__init__(ngrams=3, corpus=corpus, **kwargs)

class Quadgrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  arrow_schema = pa.schema(
    {
      'token1': pa.string(),
      'token2': pa.string(),
      'token3': pa.string(),
      'token4': pa.string(),
      'count': pa.uint32()
    }
  )
  def __init__(self, corpus, **kwargs):
    super().__init__(ngrams=3, corpus=corpus, **kwargs)

class Quintgrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  arrow_schema = pa.schema(
    {
      'token1': pa.string(),
      'token2': pa.string(),
      'token3': pa.string(),
      'token4': pa.string(),
      'token5': pa.string(),
      'count': pa.uint32()
    }
  )

  def __init__(self, corpus, **kwargs):
    super().__init__(ngrams=3, corpus=corpus, **kwargs)


class ChunkedTokenCounts(TokenCounts):
  name = "chunked_counts"
