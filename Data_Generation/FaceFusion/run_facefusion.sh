#!/bin/bash

GPU_ID=3


VIDEO_PATH=".png"
SAVE_PATH="d.png"
SOURCE_PATH="/.png"


USE_ONNX=true

if [ "$USE_ONNX" == "true" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python facefusion.py --video "$VIDEO_PATH" --savepath "$SAVE_PATH" --source "$SOURCE_PATH" --onnx
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python facefusion.py --video "$VIDEO_PATH" --savepath "$SAVE_PATH" --source "$SOURCE_PATH"
fi

