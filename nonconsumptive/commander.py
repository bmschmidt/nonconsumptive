import argparse
import sys
from pathlib import Path
from .corpus import Corpus
from .transformations import transformations
from itertools import chain
import logging
logging.getLogger("nonconsumptive").setLevel(logging.DEBUG)

def namespace_to_kwargs(namespace, additional_args = {}):
  """
  namespace: A namespace, probably from argparse.

  additional_args: additional kwargs passed directly to Corpus.
  """
  dicto = vars(namespace)
  nested = {
    'text_options': {},
    'metadata_options': {"id_field": None}
    }
  for k, v in chain(dicto.items(), additional_args.items()):
    if k.startswith("text_"):
      nested['text_options'][k.replace("text_", "")] = v
    elif k.startswith("metadata_"):
      nested['metadata_options'][k.replace("metadata_", "")] = v
    elif k == "targets":
      if v is not None:
        nested['cache_set'] = set(v)
    elif k in {'action'}:
      continue
    else:
      if k in {'texts', 'metadata', 'dir', 'only_stacks', 'batching_options', 'bookstacks'}:
        nested[k] = v
  return nested

parser = argparse.ArgumentParser(description = "Compute and cache on digital libraries",
  prog = "nonconsumptive"
)

subparsers = parser.add_subparsers(title="action",
  help='The command to run with nonconsumptive', dest="action")

build_parser = subparsers.add_parser("build", 
  description = "Cache portions of a pipeline to disk.")

validate_parser = subparsers.add_parser("validate", 
  description = "Check integrity of file inputs.")


def add_builder_parameters(build_parser):
  """
  Add some useful parameters. Broken into a function 
  because bookworm would like to borrow some of these arguments.
  """
  build_parser.add_argument("--metadata", type = Path, help = ""
  "The location of metadata for the corpus. If not passed, "
  "metadata will be inferred from the passed documents. "
  "Passed as a single file or directory. File must end with "
  "One of the following suffixes: "
  ".csv, .tsv, .xlsx, .ndjson, .feather, .parquet.")
  build_parser.add_argument("--texts", type=Path, help = ""
  "The path to the texts to be parsed.")
  build_parser.add_argument("--text-metadata-field", type = str, help = "The field in the "
  "metadata that holds full text. If passed, the corpus will be "
  "treated as one where all text information is bundled inside "
  "the metadata file.")

  build_parser.add_argument("--text-format", type = str, default="txt",
  help = ""
  "The format in which text is stored. This should "
  "correspond to the filenames if using file-based input, and may be used "
  "to change the behavior on tokenization. The latter behavior is not yet "
  "implemented. Default 'txt'",
  choices = ["txt", "md", "html", "pdf", "tei"])
  build_parser.add_argument("--metadata-id-field", type = str, help = ""
  "The field in the metadata holding a unique identifier for each text."
  "If not passed will look for a field called '@id', 'filename', or 'id'.")
  build_parser.add_argument("--dir", type = Path, help = ""
  "The directory into which to save derivative files. ",
    default=Path("nonconsumptive")
  )
  build_parser.add_argument("--only-stacks", type = str, nargs = '+', help = ""
    "Build with only a subset of the possible bookstacks. Mostly useful for distributed processing ",
    default=None
  )
  build_parser.add_argument("--bookstacks", type = Path, help = ""
  "A directory of parquet bookstacks to ingest for texts and metadata.")

  return build_parser

build_parser = add_builder_parameters(build_parser)

build_parser.add_argument("--targets", type = str, nargs='+',
  help = ""
  "The field in the metadata holding a unique identifier for each text."
  "If not passed will look for a field called '@id', 'filename', or 'id'.",
  choices = transformations)

def main(args = None):
  if args is None:
    parsed_args = parser.parse_args()
  else:
    parsed_args = parser.parse_args(args)
  corpus = Corpus(**namespace_to_kwargs(parsed_args))
  for target in parsed_args.targets:
    corpus.cache(target)
  pass