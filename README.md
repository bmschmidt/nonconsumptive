# Nonconsumptive: Make human-readable things from machine-read text.

This is a set of tools for python for working with text non-consumptively in ways that integrate 
with a broader set of tools.

1. Creating and consuming "Feature Counts" like those distributed by the Hathi Trust.
2. Counting words and feeding them in.
3. Designating and managing short portions of text to share as examples under fair use.

## Motivation

A variety of Python tools exist to read metadata ()

The Santa Barbara statement on collections as data says "<>". 

## Speed.

Working with large collections of can be slow. This library deploys the Apache Arrow format for binary storage to pass data
between processes to allow fast serialization, deserialization, and storage.

For long-term archival deposit, it will be possible to export corpora as linked open data formatted as JSON and/or XML.



## Legal disclaimer

Although the goal here is to stay within the bounds of American law, the authors and not lawyers and nothing in this document 
should be constituted as legal advice. The goal here is to follow a good-faith effort to establish and follow best practices for the distribution of features in digital humanities research, an activity which should be protected under US law. But while nonconsumptive research is protected under US law, the precise bounds have not been delimited.

## Ethical and legal claimer

Nonconsumptive reading is about avoiding legal issues involving copyright and ethical issues involving the ownership rights of authors over their texts. There are many bad things you can do with text that don't violate copyright law. 
Some of these may be illegal and unethical. (Distributing unigram counts of medical records would probably violate [HIPAA regulations](https://www.hhs.gov/hipaa/index.html). Others may be legal but unethical. (Do not build data visualizations that dox members of online communities.) Others may be ethical but illegal. (Be careful whose website you scrape.) 

