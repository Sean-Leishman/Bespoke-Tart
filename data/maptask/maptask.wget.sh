#!/bin/bash

wget -r -nH -np -A "*.txt" https://groups.inf.ed.ac.uk/maptask/transcripts/
mv maptask/transcripts transcripts

rm -rf maptask

mkdir parsed_transcripts
