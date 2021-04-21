import yaml

default = """
paths:
  text_files: texts
  metadata_file: metadata.ndjson
  feature_counts: feature_counts
  SRP: SRP

"""

user_prefs    = {}
project_prefs = {}
session_prefs = {}

default_prefs = yaml.safe_load(default)

def load_project_prefs(dir):
  global project_prefs
  pass

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