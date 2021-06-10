import pyarrow as pa
from .document import Document, tokenize, token_counts
from typing import DefaultDict, Iterator, Union, Optional, List, Tuple
from pyarrow import parquet, feather, RecordBatch 
from pyarrow import compute as pc
from .data_storage import ArrowIDChunkedReservoir, ArrowReservoir, BatchIDIndex

import logging
logger = logging.getLogger("nonconsumptive")

class Tokenization(ArrowIDChunkedReservoir):
  name = "tokenization"
  arrow_schema = pa.schema({"token": pa.string()})

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.upstream = self.corpus.texts

  def process_batch(self, input = Tuple[str, RecordBatch]) -> RecordBatch:
    id, text = input
    tokens = tokenize(text)
    return pa.record_batch([tokens], self.arrow_schema)

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

  def __init__(self, corpus, *args, **kwargs):
    self.upstream = corpus.tokenization
    self.doc_lengths = []
    super().__init__(corpus, *args, **kwargs)

  def close(self):
    self.flush_wordcounts()
    super().close()

  def flush_wordcounts(self):
    parent = self.corpus.root / "document_lengths"
    parent.mkdir(exist_ok = True)
    path = (parent / self.uuid).with_suffix(".feather")
    tab = pa.table({"count": pa.array(self.doc_lengths, pa.uint32())})
    feather.write_feather(tab, path)

  def process_batch(self, words: pa.Array) -> RecordBatch:
    words, counts = pc.value_counts(words['token']).flatten()
    self.doc_lengths.append(len(words))
    if len(self.doc_lengths) >= 2**16:
      self.flush_wordcounts()
    return pa.record_batch(
      [words, counts.cast(pa.uint32())],
      schema = self.arrow_schema  
    )

class EncodedCounts(ArrowReservoir):

  name = None

  def __init__(self, upstream: ArrowIDChunkedReservoir, *args, **kwargs):
    self.upstream = upstream
    super().__init__(*args, **kwargs)
    self.cache = []
  
  def flush(self, wordids) -> pa.Table:
    tb = pa.Table.from_batches(self.cache)
    derived = {}
    for name in tb.schema.names:
      if name == "token":
        derived['wordid'] = pc.index_in(tb[name], value_set = wordids).cast(pa.uint32())
      elif name.startswith("word"):
        derived[name] = pc.index_in(tb[name], wordids).cast(pa.uint32())
      else:
        derived[name] = tb[name] 
    return pa.table(derived)

  def _from_upstream(self) -> pa.Table:
    wordids = self.corpus.total_wordcounts["token"]
    cache_size = 0
    for batch in self.upstream.iter_with_ids("_ncid"):
      self.cache.append(batch)
      cache_size += batch.nbytes
      if cache_size > 5_000_000:
        for batch in self.flush(wordids).to_batches():
          yield batch
        cache_size = 0
    for batch in self.flush(wordids).to_batches():
      yield batch

class EncodedUnigrams(EncodedCounts):
  name = "ncid_wordid"
  def __init__(self, corpus, *args, **kwargs):
    super().__init__(corpus.token_counts, corpus)

class Ngrams(ArrowIDChunkedReservoir):
  def __init__(self, ngrams: int, corpus: "Corpus", end_chars: List[str] = [], beginning_chars: List[str] = [], **kwargs):
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
    super().__init__(corpus, **kwargs)


  def _from_local(self):
    pass

  @property
  def arrow_schema(self) -> pa.Schema:
    if self._arrow_schema:
      return self._arrow_schema
    labs = {}
    for n in range(ngrams):
      labs[f"token{n + 1}"] = pa.utf8()
      labs['count'] = pa.uint32()
    self._arrow_schema = pa.schema(labs)
    return self._arrow_schema
  
  def _from_upstream(self) -> Iterator[RecordBatch]:
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

transformations = {
  'token_counts': TokenCounts,
  'tokenization': Tokenization,
  'quintgrams': Quintgrams,
  'bigrams': Bigrams,
  'trigrams': Trigrams,
  'quadgrams': Quadgrams,
  'encoded_wordcounts': EncodedUnigrams
}