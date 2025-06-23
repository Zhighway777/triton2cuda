#!/bin/bash

# 获取目录名参数，默认为example_submission
DIR_NAME=${1:-example_submission}
cd ../$DIR_NAME/

# 创建压缩包，不包含example_submission目录
rm -f ../$DIR_NAME.zip
zip -r ../$DIR_NAME.zip . -x "*.pyc" "__init__.py" "__pycache__/*"