#!/bin/bash
cd ..
export PYTHONPATH=$PWD/:$PYTHONPATH
python tools/test.py \
  /home/mayson/Desktop/MACP/macp_opv2v_epoch_20.py /home/mayson/Desktop/MACP/macp_opv2v_epoch_20.pth
