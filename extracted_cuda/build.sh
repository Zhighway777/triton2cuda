#!/bin/bash
# CUDA编译脚本

echo "编译CUDA程序..."

echo "编译 matmul_temp.cu..."
nvcc -O3 -arch=sm_60 -o matmul_temp matmul_temp.cu -lcudart
if [ $? -eq 0 ]; then
    echo "  -> matmul_temp 编译成功"
else
    echo "  -> matmul_temp 编译失败"
fi

echo "编译 vecadd_temp.cu..."
nvcc -O3 -arch=sm_60 -o vecadd_temp vecadd_temp.cu -lcudart
if [ $? -eq 0 ]; then
    echo "  -> vecadd_temp 编译成功"
else
    echo "  -> vecadd_temp 编译失败"
fi

echo "编译 layer_norm_temp.cu..."
nvcc -O3 -arch=sm_60 -o layer_norm_temp layer_norm_temp.cu -lcudart
if [ $? -eq 0 ]; then
    echo "  -> layer_norm_temp 编译成功"
else
    echo "  -> layer_norm_temp 编译失败"
fi

echo "编译 mat_transpose_temp.cu..."
nvcc -O3 -arch=sm_60 -o mat_transpose_temp mat_transpose_temp.cu -lcudart
if [ $? -eq 0 ]; then
    echo "  -> mat_transpose_temp 编译成功"
else
    echo "  -> mat_transpose_temp 编译失败"
fi

echo "编译 softmax_temp.cu..."
nvcc -O3 -arch=sm_60 -o softmax_temp softmax_temp.cu -lcudart
if [ $? -eq 0 ]; then
    echo "  -> softmax_temp 编译成功"
else
    echo "  -> softmax_temp 编译失败"
fi

echo "编译完成！"