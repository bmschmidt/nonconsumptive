import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
import pyarrow as pa
import logging


class TestCaching():  
    def test_creates_intermediate(self, tmpdir):
        def number_of_files(corpus):
            counts = []
            tb = pa.Table.from_batches([*corpus.token_counts()])
            assert len(tb) == 7
            ipcs = [*Path(tmpdir).glob("**/*.feather")]
            return len(ipcs)

        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {}, text_options = {"format" : "md", "compression" : None})
        baseline = number_of_files(corpus)
        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {"tokenization"}, text_options = {"format" : "md", "compression" : None})

        assert number_of_files(corpus) - baseline == 1
        # Reiterate with same item.
        # Should not create a new file.


        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {"tokenization"}, text_options = {"format" : "md", "compression" : None})

        assert number_of_files(corpus) - baseline == 1

        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {"token_counts", "tokenization"}, text_options = {"format" : "md", "compression" : None})
        assert number_of_files(corpus) - baseline == 2

