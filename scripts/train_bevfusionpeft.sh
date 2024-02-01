#!/bin/bash
cd ..
export PYTHONPATH=$PWD/:$PYTHONPATH
python tools/train.py \
  projects/Coperception/configs/bf_peft_v2v4real.py
