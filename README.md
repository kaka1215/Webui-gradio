# Webui-gradio

记录多模态大模型评测系统的Webui界面开发，**仅包含ui相关代码（gradio）**。

## 🌲代码结构

```plaintext
Webui-gradio/
├── test.py /ui interface
├── requirements.txt
└── README.md
```

## 🚀快速开始

### 1.安装依赖

```shell
pip install -r requirements.txt
```

### 2.修改相应配置

```python
# ==== 配置shell文件 ====
MODEL_SCRIPT_DIR = ""
ATTACK_SCRIPT_DIR = ""
ATTACK_SCRIPT_PATH = ""
RESULT_PATTERNS = {}

# ===== 固定映射：模型名称 → 脚本文件名 =====
MODEL_SCRIPTS_MAP = {}

# ===== 固定映射：攻击算法名称 → 脚本文件名 =====
ATTACK_SCRIPTS_MAP = {}

# ===== 全局：任务/数据集 → 合法 infer/eval 搭配（通用于所有评测）=====
CONFIG_MATRIX = {}

# ===== 评测指标运行代码 ====
command = ()
```

### 3.运行ui挂载代码

```shell
python test.py
```

