# Nonconsumptive: Machine-read text

Rich collections of digital text consist of text and metadata.

This package provides a fast, simple set of tools for sharing stacks of text--
not just single document, but full-on libraries and collections including
as much textual information as you're willing to share.

It's designed to serve two different audiences.

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

To quote the [Santa Barbara statement on collections as data](https://collectionsasdata.github.io/statement/):
> By conceiving of, packaging, and making collections available as data,
 cultural heritage institutions work to expand the set of possible opportunities for engaging with collections.

The more abstract and computer-friendly the ways a collection is distributed,
ironically, the more different channels for human engagement it can have.

I've built this right now largely to replace a backend for one specific purpose;
serving interactive visualizations of wordcounts. But as the feature counts
distributed the HathiTrust research center have shown, these kinds of nonconsumptive
representations can have all sorts of uses.

## Formats

Humanists tend to fetishize unicode-formatted `.txt` files, often distributing
files as JSON. Internally, this library is fast because it uses instead the
emerging standards for batched, columnar binary data processing from the Apache
foundation, "Arrow" and "Parquet."

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

* **Differential disclosure.** Open is great, closed is great, halfway open is great. If you want to start a project off only metadata disclosure, then scale it up to vectorized features
with your first publication, and finish up with counts of trigrams after the book
comes out; great. Take your time. We'll still be here.
* **Format flexibility.** There are a lot of ways that make sense to distribute a collection of texts. Some people do it is
as a giant text file with tabs. Some people have zip files. Some people
have TEI with insanely well marked up paragraphs. All of these are appropriate
for different purposes; all allow different sorts of representation.
* **Metadata is as important as data** Machine learning frameworks for text
tend to treat corpora as texts alone, but reasonably organized metadata from
the library world are just as essential a component as well tokenized text.
Linked open data, on the other hand, is usually distributed in forms that
researchers ignore because it scales so poorly to columnar representations. The
solution is to use modern binary interchange formats--not CSVs--that allow
fuller representation of the complex data models information science has
given us in the last decades.
* **Minimalist interoperability.** All texts in Unicode are kind of similar. We aim to provide minimal encodings to a variety of forms--from sparse term-document matrices, raw tokenizations,
to information-conserving bigrams.
* **Information accounting.** Nonconsumptive data actually gives away a little information. People should be able to understand the type and amount of information they're distributing, so that they feel OK about doing it.
* **English language fifth.** All operations take place on Unicode here. No
flexibility for your old Latin-1 files. But also, let's take advantage of
the fact that lots of humanistic data doesn't fit so great into that
language model you trained on Wikipedia, and come up with context-agnostic
representations of language. No features should be added that specifically
target English if they don't exist for four other languages--including at least
one of Arabic, Chinese, Hindi, Russian, or Turkish--first.

## Speed.

Working with large collections of can be slow.
This library deploys the Apache Arrow format for binary storage to pass data
between processes to allow fast serialization, deserialization, and storage.

Different file formats used for different purposes.

1. All *internal* interchange--including caches--uses the IPC Apache Arrow
   format (AKA, "feather").
2. All *external* interchange uses either Arrow (when appropriate) or Apache
   Parquet, which has some advantages as an on-disk serialization format.
3. I aspire to having it be possible to export corpora as linked open data
   formatted as JSON and/or XML.

### Caching

In general, caching helps speed on these tasks, although you can take it too far.
The default arguments will cache token counts of various forms. But it's
possible to stream straight into a binary-encoded format if you like, and
it's also possible to cache the tokenizations which may make bigram and trigram
creation easier.

## Code strategies

All internal data representations should use the Apache Arrow file format
for internal transfer and parquet for storage.
These formats preserve types and allow binding of metadata to tables, and so present a much more efficient way of storing data. Export to CSV and/or JSON for archival purposes will be supported on the condition that you say out loud three times the phrase "JSON is a fine archival format, but it's a lousy interchange format."

Internal methods like `tokenization` and `bigram_counts` are exposed in general
as an iterator over Apache Arrow record batches. There are some convenience
methods to access individual books by ids, but for large-scale throughput it
makes sense to iterate over the documents in an arbitrary order.

I don't especially like pandas, so all code has to work with Arrow table instead.
For now more complicated operations like joins are handled by the `polars`
library, but those internal implementations may change.

### Contributor rules

1. No pandas imports outside of the 'engine'.
2. All functions should use python type annotations.
3. Unit tests are good, working examples in ipynb are good. Tests use `pytest.`

## Legal disclaimer

Although the goal here is to stay within the bounds of American law, the authors
are not lawyers and nothing in this document or package.
should be constructed as legal advice. The goal here is to offer a good-faith
effort to establish and follow best practices for the distribution of
features in digital humanities research, an activity which should be protected
under US law. But while nonconsumptive research is protected under US law,
the precise bounds have not been delimited.

## Ethical and legal claimer

Nonconsumptive reading is about avoiding legal issues involving copyright and
ethical issues involving the ownership rights of authors over their texts.
There are many bad things you can do with text that don't violate copyright law.
Some of these may be illegal and unethical. (Distributing unigram counts of
medical records would probably violate
[HIPAA regulations](https://www.hhs.gov/hipaa/index.html).
Others may be legal but unethical. (Do not build data visualizations that dox
members of online communities.) Others may be ethical but illegal.
(Be careful whose website you scrape.) I don't really know what to tell you
to do. I generally don't think we have ethical obligations towards the dead,
but what do I know?

> Every man’s conscience is vile and depraved
> You cannot depend on it to be your guide
> When it’s you who must keep it satisfied.
