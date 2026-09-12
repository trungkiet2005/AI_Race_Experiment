# ARR paper build directory

Self-contained. Nothing here reaches outside this folder, and nothing outside
this folder should reach in. In particular this directory does **not** share a
bibliography, a class file or a build script with the manuscript one level up:
that manuscript is a concurrent submission to a different venue and is being
rewritten by another workflow.

## What is here

| file | origin |
|---|---|
| `acl.sty` | official ACL style files, unmodified |
| `acl_natbib.bst` | official ACL style files, unmodified |
| `acl_latex.tex` | official ACL sample document, unmodified, kept for reference |
| `acl_lualatex.tex` | official ACL sample for lua/xelatex, unmodified |
| `custom.bib` | this paper's bibliography, one entry: the concurrent submission |
| `anthology.bib.txt` | the upstream note explaining where the Anthology bibliography is obtained; it is **not** a bib file and must not be renamed to one |
| `main.tex` | the paper |

Download source and checksums for the template files are recorded in
`docs/acl-audit-paper-plan.md`, not in this directory, because nothing under
`paper/` may carry a repository URL.

## Status

**Not submittable.** The evidence sections are written and every number in
them was recomputed from the artifact named in the comment block above it. The
build is clean: no LaTeX errors, no overfull boxes, no undefined references or
citations, no Type 3 fonts, and the anonymity scan passes. Body content
through the conclusion fits inside the eight-page limit with room to spare.

Three things are missing and the document says so where a reader would
otherwise assume otherwise.

1. **A roster where the endpoint name and the comprehension score disagree.**
   The paper's central result is that the battery's verdict is reproduced by a
   regular expression over the endpoint name on all nine endpoints. Until one
   endpoint breaks that tie, the battery cannot be shown to measure
   comprehension rather than vendor size tier. Marked in the results section
   and in the limitations.
2. **Frozen probe banks for two further task families**, an iterated social
   dilemma and a public-goods or common-pool game, administered to the same
   roster. Without them the transfer section is empty and the paper measures
   one task rather than a construct. Marked in the task-families section and
   in the transfer section, both of which carry a visible
   `\pending{...}` block.
3. **Related work.** Four strands are named in the source comments and not one
   of their citations has been verified at the source in this repository, so
   the section is left unwritten rather than filled with unverified entries.

`custom.bib` holds a single entry, the anonymous concurrent submission. The
ACL example entries that shipped with the template have been deleted, and so
has the `\nocite{*}` that existed only to exercise the bibliography style.

## Build

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Two passes after bibtex, for references and cross-references. Do not use
`scripts/build_publication.py`: it builds the other manuscript, and it is not
safe to run while figures are being regenerated.

The `[review]` option in the preamble stays until acceptance. ARR review is
anonymous.

## Checks to run before this file goes anywhere

```
python scripts/check_paper_crossrefs.py     # a \ref must resolve inside its own document
python scripts/check_house_style.py         # punctuation dashes, truncated sentences
python scripts/anonymity_scan.py paper/acl/main.pdf
```

`check_paper_crossrefs.py` exists because a `\ref` from one of these documents
to a label defined only in the other does not resolve and prints as `??`. Each
document is built on its own.

## Rules for anything written into this directory

No author name, no institution, no account handle, no repository URL, and not
the name of the project directory either. `scripts/anonymity_scan.py` refuses
all of those; run it against the built PDF before sending the file anywhere.

Do not paste prose across from the concurrent manuscript. The two papers share
a roster, a game and a collection protocol, and the only thing that makes this
a separate contribution is that its unit of analysis is the screen rather than
the endpoint. A section assembled from the other paper's sentences reads as
offcuts, and ARR desk-rejects a thinly sliced contribution.
