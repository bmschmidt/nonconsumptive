import pytest
from pathlib import Path
import pandas as pd
from nonconsumptive import Corpus
import pyarrow as pa

@pytest.fixture(scope="module")
def test_corpus():
    return Corpus(Path('tests', 'corpora', 'minicomp_ed'))

@pytest.fixture(scope="session")
def image_file(tmpdir_factory):
    img = compute_expensive_image()
    fn = tmpdir_factory.mktemp("data").join("img.png")
    img.save(str(fn))
    return fn

class TestCaching():
    @classmethod
    def setup_class(cls):
        for p in Path("tests/corpora/minicomp_ed").glob("**/*.ipc"):
            p.unlink()
    
    def test_creates_intermediate(self):
        corpus = Corpus(Path('tests', 'corpora', 'minicomp_ed'), cache_set = {"tokenization"}, format = "md", compression = None)
        counts = []
        for count in corpus.token_counts:
            counts.append(count)
        assert len(counts) == 7
        ipcs = [*Path("tests", 'corpora', "minicomp_ed").glob("**/*.ipc")]
        assert(len(ipcs) == 1)

        counts = []
        # Reiterate with same item.
        for count in corpus.token_counts:
            counts.append(count)
        ipcs = [*Path("tests", 'corpora', "minicomp_ed").glob("**/*.ipc")]
        assert(len(ipcs) == 1)

        corpus = Corpus(Path('tests', 'corpora', 'minicomp_ed'), cache_set = {"token_counts", "tokenization"}, format = "md", compression = None)

        counts = []
        for count in corpus.token_counts:
            counts.append(count)
        assert len(counts) == 7
        ipcs = [*Path("tests", 'corpora', "minicomp_ed").glob("**/*.ipc")]
        assert(len(ipcs) == 2)
