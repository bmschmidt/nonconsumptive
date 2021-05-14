import yaml

default = """
paths:
  text_files: texts
  metadata_file: null
  feature_counts: feature_counts
  wordids: wordids
  SRP: SRP
cache:
#  - feature_counts
  - wordids
  - token_counts
"""

user_prefs    = {}
project_prefs = {}
session_prefs = {}

default_prefs = yaml.safe_load(default)

def load_project_prefs(dir):
  global project_prefs
  pass

def set_prefs(key, value):
  global session_prefs
  components = key.split(".")
  p = session_prefs
  for i, key in enumerate(components):
    if i == len(components) - 1:
      p[key] = value
    else:
      try:
        p = p[key]
      except KeyError:
        p[key] = {}
        p = p[key]


def prefs(key):
  components = key.split(".")
  for pref_set in [session_prefs, project_prefs, user_prefs, default_prefs]:
    for i, key in enumerate(components):
      try:
        pref_set = pref_set[key]
      except KeyError:
        break
      if i == len(components) - 1:
        return pref_set 
  raise KeyError("Unable to find prefs")