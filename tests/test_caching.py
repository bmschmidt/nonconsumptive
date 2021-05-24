import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
import pyarrow as pa

class TestCaching():
    @classmethod
    def setup_class(cls):
        for p in Path("tests/corpora/minicomp_ed").glob("**/*.ipc"):
            p.unlink()
        for p in Path("tests/corpora/minicomp_ed").glob("**/*.json"):
            p.unlink()
    
    def test_creates_intermediate(self, tmpdir):
        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {"tokenization"}, text_options = {"format" : "md", "compression" : None})
        counts = []
        for count in corpus.token_counts:
            counts.append(count)
        assert len(counts) == 7
        ipcs = [*Path(tmpdir).glob("**/*.ipc")]
        print(ipcs)

        assert(len(ipcs) == 1)

        counts = []
        # Reiterate with same item.
        # Should not create anything new.
        for count in corpus.token_counts:
            counts.append(count)
        ipcs = [*Path(tmpdir).glob("**/*.ipc")]
        print(ipcs)
        assert(len(ipcs) == 1)

        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {"tokenization"}, format = "md", compression = None)

        corpus = Corpus(texts = Path('tests', 'corpora', 'minicomp_ed'), dir = tmpdir,
            cache_set = {"token_counts", "tokenization"}, format = "md", compression = None)

        counts = []
        for count in corpus.token_counts:
            counts.append(count)

        assert len(counts) == 7
        ipcs = [*Path(tmpdir).glob("**/*.ipc")]
        assert(len(ipcs) == 2)
