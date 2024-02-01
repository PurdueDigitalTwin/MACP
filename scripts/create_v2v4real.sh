#!/bin/bash
cd .. || exit

python projects/Coperception/tools/create_data.py \
  --dataset v2v4real \
  --root-path data/v2v4real \
  --cache-path data/v2v4real/cache/ef \
  --version train \
  --info-prefix v2v4real \
  --num_workers 4

python projects/Coperception/tools/create_data.py \
  --dataset v2v4real \
  --root-path data/v2v4real \
  --cache-path data/v2v4real/cache/ef \
  --version test \
  --info-prefix v2v4real \
  --num_workers 4
