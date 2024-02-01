#!/bin/bash
cd .. || exit

python projects/Coperception/tools/create_data.py \
  --dataset openv2v \
  --root-path data/openv2v \
  --cache-path data/openv2v/cache/ef \
  --version train \
  --info-prefix openv2v \
  --num_workers 4

python projects/Coperception/tools/create_data.py \
  --dataset openv2v \
  --root-path data/openv2v \
  --cache-path data/openv2v/cache/ef \
  --version validate \
  --info-prefix openv2v \
  --num_workers 4

python projects/Coperception/tools/create_data.py \
  --dataset openv2v \
  --root-path data/openv2v \
  --cache-path data/openv2v/cache/ef \
  --version test \
  --info-prefix openv2v \
  --num_workers 4

python projects/Coperception/tools/create_data.py \
  --dataset openv2v \
  --root-path data/openv2v \
  --cache-path data/openv2v/cache/ef \
  --version test_culver_city \
  --info-prefix openv2v \
  --num_workers 4
