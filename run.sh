#!/usr/bin/env sh
python SCNN_generator.py \
    --height 46 \
    --width 46 \
    --kernel_width 9 \
    --channel 128 \
    --bottom conv5_5 \
    --output SCNN.prototxt
