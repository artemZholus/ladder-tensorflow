#!/usr/bin/env bash
mkdir -p experiments
mkdir -p stat
python3 benchmark.py | tee summary.txt