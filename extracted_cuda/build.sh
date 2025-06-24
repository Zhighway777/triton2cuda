#!/bin/bash
# CUDA编译脚本

echo "编译CUDA程序..."

echo "编译 vecadd.cu..."
nvcc -O3 -arch=sm_60 -o vecadd vecadd.cu -lcudart
if [ $? -eq 0 ]; then
    echo "  -> vecadd 编译成功"
else
    echo "  -> vecadd 编译失败"
fi

echo "编译完成！"