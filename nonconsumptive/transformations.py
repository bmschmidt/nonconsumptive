import pyarrow as pa
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  import nonconsumptive as nc
from pyarrow import parquet, feather, RecordBatch 
from pyarrow import compute as pc
from pyarrow import ipc
from .data_storage import ArrowIDChunkedReservoir, ArrowLineChunkedReservoir, ArrowReservoir
import logging
logger = logging.getLogger("nonconsumptive")
import polars as pl

try:
  import blingfire
  def tokenize(text):
    return blingfire.text_to_words(text).split(" ")
except:
  logger.warning("Couldn't find blingfire, falling back to regex tokenization. `pip install blingfire` for faster tokenization.")
  from .document import tokenize

class Tokenization(ArrowIDChunkedReservoir):
  name = "tokenization"
  arrow_schema = pa.schema({"token": pa.string()})

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def upstream(self):
    ids = self.bookstack.ids['@id']
    yield from self.corpus.text_input.iter_texts_for_ids(ids)
  
  def process_batch(self, text = str) -> RecordBatch:
    tokens = tokenize(text)
    return pa.record_batch([tokens], self.arrow_schema)

class SRP_Transform(ArrowLineChunkedReservoir):
  name = "srp_transform"

class DocumentLengths(ArrowLineChunkedReservoir):
  name = "document_lengths"
  arrow_schema = pa.schema({"nwords": pa.uint32()})

  def _from_upstream(self) -> Iterator[pa.RecordBatch]:
    counts = []
    for batch in self.bookstack.get_transform("token_counts"):
      counts.append(pc.sum(batch['count']).as_py())
      if len(counts) >= 2**16:
        yield pa.record_batch([pa.array(counts, pa.uint32())], names = ['nwords'])
        counts = []
    if len(counts):
      yield pa.record_batch([pa.array(counts, pa.uint32())], names = ['nwords'])

class TokenCounts(ArrowIDChunkedReservoir):
  """
  
  A TokenCounts objects caches counts by unigram for 
  each document. It also build up a total list of
  document lengths for all documents.

  """
  name = "token_counts"
  arrow_schema = pa.schema({
    'token': pa.string(),
    'count': pa.uint32()
  })

  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack, *args, **kwargs)
    self._upstream = self.bookstack.get_transform("tokenization")

  def close(self):
    super().close()

  def process_batch(self, words: pa.Array) -> RecordBatch:
    words, counts = pc.value_counts(words['token']).flatten()
    return pa.record_batch(
      [words, counts.cast(pa.uint32())],
      schema = self.arrow_schema  
    )

class EncodedCounts(ArrowReservoir):

  def __init__(self, upstream: ArrowIDChunkedReservoir, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._upstream = upstream
    self.cache = []
  
  @property
  def wordids(self) -> pa.Array:
    return self.corpus.total_wordcounts["token"]

  def flush(self) -> pa.Table:
    if len(self.cache) == 0:
      return
    tb = pa.Table.from_batches(self.cache)
    derived = {}
    for name in tb.schema.names:
      if name == "token":
        derived['wordid'] = pc.index_in(tb[name], value_set = self.wordids).cast(pa.uint32())
      elif name.startswith("word"):
        derived[name] = pc.index_in(tb[name], self.wordids).cast(pa.uint32())
      else:
        derived[name] = tb[name]
    self.cache = []
    return pa.table(derived)

  def _from_upstream(self) -> pa.Table:
    """
    Rather than work a batch at a time, try to do at least 10MB at once,
    so the join can be a bit more efficient.
    """
    cache_size = 0
    for batch in self.bookstack.get_transform("token_counts").iter_with_ids("_ncid"):
      self.cache.append(batch)
      cache_size += batch.nbytes
      if cache_size > 10_000_000:
        yield from self.flush().to_batches()
        cache_size = 0
    if len(self.cache):
      yield from self.flush().to_batches()

class EncodedUnigrams(EncodedCounts):
  name : str = "ncid_wordid"
  def __init__(self, bookstack, *args, **kwargs):
    super().__init__(bookstack.get_transform("token_counts"), bookstack)

class Ngrams(ArrowIDChunkedReservoir):
  def __init__(self, bookstack, ngrams: int, end_chars: List[str] = [], beginning_chars: List[str] = [], **kwargs):
    """

    Creates an (optionally cached) iterator over record batches of ngram counts.
    Each batch is a single document, with columns ['token1', 'token2', ... , 'token{n}', count]

    ngrams: an integer. Size of the ngrams to construct.
    end_chars: a list of regular expression (re2 compatible) to treat as the *end* of an n-gram.
               For instance, [r"[\\.\\?\\!]"] would attach sentence-ending punctuators to the end of
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
    self._arrow_schema = None
    super().__init__(bookstack, **kwargs)

  def _from_local(self):
    pass

  @property
  def arrow_schema(self) -> pa.Schema:
    if self._arrow_schema:
      return self._arrow_schema
    labs = {}
    for n in range(self.ngrams):
      labs[f"token{n + 1}"] = pa.utf8()
      labs['count'] = pa.uint32()
    self._arrow_schema = pa.schema(labs)
    return self._arrow_schema
  
  def _from_upstream(self) -> Iterator[RecordBatch]:
    ngrams = self.ngrams
    for batch in self.bookstack.tokenization():
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
        empty : List = [[]]
        # Still register that we saw the id even if it's not long enough for ngrams
        yield pa.RecordBatch.from_arrays(empty * len(self.arrow_schema.names), schema=self.arrow_schema)
        continue
      # Temporarily pass through pandas because https://github.com/pola-rs/polars/issues/668
      t = pl.from_pandas(ngram_tab.to_pandas())
      counts = t.groupby([f"token{i+1}" for i in range(ngrams)])\
          .agg([pl.count("token1").alias("count")])

      other_cols = []
      for i in range(ngrams):
          other_cols.append(counts[f'token{i+1}'].to_arrow().cast(pa.string()))
      other_cols.append(counts['count'].to_arrow())
      yield pa.RecordBatch.from_arrays(other_cols, schema=self.arrow_schema)


class Bigrams(Ngrams):
  """
  Convenience around Ngrams for the case of n==2.
  """
  name = "bigrams"

  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=2, **kwargs)

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
  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=3, **kwargs)

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

  def __init__(self, bookstack, **kwargs):
    super().__init__(bookstack, ngrams=3, **kwargs)

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

transformations = {
  'document_lengths': DocumentLengths,
  'token_counts': TokenCounts,
  'tokenization': Tokenization,
  'quintgrams': Quintgrams,
  'bigrams': Bigrams,
  'trigrams': Trigrams,
  'quadgrams': Quadgrams,
  'encoded_wordcounts': EncodedUnigrams
#  'encoded_bigrams': EncodedBigrams
}

