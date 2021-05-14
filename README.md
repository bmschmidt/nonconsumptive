# Nonconsumptive: Machine-read text

Rich collections of digital text consist of text and metadata.

This package provides a fast, simple set of tools for sharing and reading texts
nonconsumptively.

1. Researchers who want to *work* with nonconsumptive data, either as their
   only source or as a supplement to other methods.
2. Scholars, librarians, and others who want *distribute* information about texts
   in their collections nonconsumptively.



Some of the tasks this library is designed to support include:

1. Create and/or consume "Feature Counts" like those distributed by the
   Hathi Trust with appropriate metadata for computational analysis.
2. Convert among different formats for corpora. (Create basic TEI from a
   CSV of metadata).
3. Create rich web visualizations from textual corpora.
4. Designating and managing short portions of text to share as examples
   under fair use.

## Motivation


The Santa Barbara statement on collections as data says "<>".


## Flow

The basic principle is to describe the flow of data from two sources--data and metadata--into
something that be analyzed and distributed as feature counts.

```
text -> tokenization -> feature_counts -> feature_counts -> encoded_features.--\
                                      \                                         Interfaces
                                       feature_counts_with_metadata            /
metadata -> normalized_metadata ----- / --------------------------------------/
schema ----/
```

## Principles

* **Differential disclosure.** Open is great, closed is great, halfway open is great. If you want to start a project off
with.
* **Format flexibility.** There are a lot of ways that make sense to distribute a collection of texts. Some people do it is
as a giant text file with
* **Metadata is as important as data** Machine learning frameworks for text tend to treat corpora as texts alone, but reasonably organized metadata from the library world are just as essential a component
* **Minimalist interoperability.** All texts in Unicode are kind of similar. We aim to provide minimal encodings to a variety of forms--from sparse term-document matrices.
* **Information accounting.** Nonconsumptive data actually gives away a little information. People should be able to understand the type and amount of information they're distributing, so that they feel OK about doing it.
* **English language fifth.** All operations take place on unicode, and no features should be added solely to support
English if they don't exist four other languages first.

## Speed.

Working with large collections of can be slow.
This library deploys the Apache Arrow format for binary storage to pass data
between processes to allow fast serialization, deserialization, and storage.

For long-term archival deposit, it will be possible to export corpora as linked open data formatted as JSON and/or XML.

### Caching

In general, caching helps speed on these tasks, although you can take it too far.

## Code strategies

All internal data representations should use the Apache Arrow file format for internal transfer and parquet for storage.
These formats preserve types and allow binding of metadata to tables, and so present a much more efficient way of storing data. Export to CSV and/or JSON for archival purposes will be supported on the condition that you say out loud three times the phrase "JSON is a fine archival format, but it's a lousy interchange format."

### Contributor rules

1. No pandas imports outside of the 'engine'
2. All functions should use python type annotations.
3. Unit tests are good, working examples in ipynb are good.

## Legal disclaimer

Although the goal here is to stay within the bounds of American law, the authors and not lawyers and nothing in this document
should be constituted as legal advice. The goal here is to follow a good-faith effort to establish and follow best practices for the distribution of features in digital humanities research, an activity which should be protected under US law. But while nonconsumptive research is protected under US law, the precise bounds have not been delimited.

## Ethical and legal claimer

Nonconsumptive reading is about avoiding legal issues involving copyright and ethical issues involving the ownership rights of authors over their texts. There are many bad things you can do with text that don't violate copyright law.
Some of these may be illegal and unethical. (Distributing unigram counts of medical records would probably violate [HIPAA regulations](https://www.hhs.gov/hipaa/index.html). Others may be legal but unethical. (Do not build data visualizations that dox members of online communities.) Others may be ethical but illegal. (Be careful whose website you scrape.)
