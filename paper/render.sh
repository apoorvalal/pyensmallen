#!/bin/bash

pandoc paper.md -o paper.pdf \
  --pdf-engine=xelatex \
  --bibliography=paper.bib \
  --citeproc \
  --number-sections
