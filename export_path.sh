#!/bin/bash

# 获取 .env 文件的目录路径
export ENV_PATH=$(dirname "$(realpath .env)")


# 输出 ENV_PATH，确保路径已正确加载
echo "ENV_PATH is set to: $ENV_PATH"

# 可选：加载 .env 文件中的内容到环境变量
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# 可选：将 ENV_PATH 添加到 PYTHONPATH（如果需要）
export PYTHONPATH="$PYTHONPATH:$ENV_PATH"


