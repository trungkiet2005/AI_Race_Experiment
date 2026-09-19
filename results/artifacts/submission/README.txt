AAMAS 2027 submission files
===========================

Upload to the portal:
  paper.pdf          the manuscript, anonymous
  supplementary.zip  supplementary material, single zip as the portal requires:
                     supplementary.pdf plus code/, the anonymous code snapshot
                     (code/README.md says how to run the claim checker)

Not uploaded, here for reading only:
  supplementary.pdf  the same document the zip contains
  *.anonymity.json   the double-blind scan verdict for each file beside it,
                     keyed to that file's SHA-256. Upload neither of these.

Both PDFs build from paper/main.tex and paper/supplementary.tex through
scripts/build_publication.py, which runs the same anonymity scan at the
moment each PDF is made. Rebuild them before rebuilding this bundle.
code/ is results/_build/code_snapshot, built by scripts/build_code_snapshot.py.

Before the camera-ready, switch main.tex back to the non-anonymous
\documentclass[sigconf]{aamas} and restore the acknowledgements block
that sits commented out near the end of the same file.
