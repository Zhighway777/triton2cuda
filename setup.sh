#!/bin/bash

# Triton2CUDA 最简环境配置
set -e

echo "🚀 配置Triton2CUDA环境..."

# 检查项目目录
if [ ! -f "requirements.txt" ]; then
    echo "❌ 请在项目根目录运行"
    exit 1
fi

# 安装系统依赖
echo "📦 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y build-essential ninja-build

# 安装CUDA工具包
if ! command -v nvcc &> /dev/null; then
    echo "🔧 安装CUDA 12.4..."
    if [ ! -f "/etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list" ]; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update -qq
        rm -f cuda-keyring_1.0-1_all.deb
    fi
    sudo apt-get install -y cuda-toolkit-12-4
fi

# 设置CUDA环境
if [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 创建虚拟环境
VENV_NAME="triton2cuda-env"
echo "📦 创建虚拟环境..."
[ -d "$VENV_NAME" ] && rm -rf "$VENV_NAME"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"
pip install --upgrade pip

# 安装核心包
echo "🔥 安装PyTorch..."
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "💫 安装Triton..."
pip install triton==3.0.0

# 安装requirements.txt中的依赖
echo "📋 安装其他依赖..."
while IFS= read -r line; do
    [[ $line =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]] && continue
    if [[ $line =~ ^pip[[:space:]]+install ]]; then
        [[ $line =~ torch== ]] && continue
        eval "$line"
    else
        [[ $line =~ triton== ]] && continue
        pip install "$line"
    fi
done < requirements.txt

# 保存CUDA环境变量
cat >> "$VENV_NAME/bin/activate" << EOF

# CUDA环境变量
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF

# 创建激活脚本
cat > activate_env.sh << EOF
#!/bin/bash
source "$PWD/$VENV_NAME/bin/activate"
echo "✅ 环境已激活"
EOF
chmod +x activate_env.sh

echo ""
echo "🎉 配置完成！"
echo "运行 ./activate_env.sh 激活环境"
