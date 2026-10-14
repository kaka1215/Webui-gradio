<div align="center">

# EvalDeck

**多模态模型评测工作台**

一个界面，管理模型服务与六类评测。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#快速开始)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-F97316)](https://github.com/gradio-app/gradio)
[![License: MIT](https://img.shields.io/badge/License-MIT-2563EB)](LICENSE)

[English](README.md) · 简体中文

</div>

通过 Gradio 界面管理模型服务、配置评测任务、查看实时日志和结果。项目最初开发于 2025 年，目前已将配置、界面、执行和结果处理拆分为独立模块。

本仓库提供**界面与任务编排代码**。模型权重、数据集、攻击实现与评测后端需要自行接入。

![六类评测与工作流程](docs/assets/overview.svg)

## 六类评测

| 维度 | 配置内容 |
| --- | --- |
| **准确性** | 任务、数据集、推理/评测搭配与模型服务 |
| **抗干扰性** | 椒盐噪声、高斯噪声、运动模糊、JPEG 压缩和缩放 |
| **抗攻击性** | 攻击方式、数据集对应的评测指标与攻击参数 |
| **泛化性** | 两个数据集及各自的推理/评测配置 |
| **运行效率** | 效率评测所需的数据集与模型配置 |
| **防御** | 攻击、防御参数及辅助模型端口 |

具体指标计算和变换由接入的后端实现。

## 功能

- **标签页工作台**：六类评测使用统一的运行与结果控件。
- **YAML 配置接入**：服务脚本、任务组合、执行命令和结果路径均无需修改 Python 代码。
- **配置预览**：运行前校验并查看 YAML，预览中隐藏密钥和代理信息。
- **进程管理**：指定 GPU、查看实时输出、停止运行中的任务，并自动清理临时配置。
- **统一结果展示**：六类评测均支持 JSON 展示、大文件预览和下载，不会将旧结果显示为本次结果。
- **回归检查**：使用临时 Python 后端验证配置和执行流程，不需要模型或 GPU。

<details open>
<summary>界面预览</summary>

![评测工作台](docs/assets/evaluation.png)

截图展示示例任务配置的 YAML 预览，不包含模拟的评测分数。

</details>

## 快速开始

使用 Python 3.10 或更高版本：

```bash
git clone https://github.com/kaka1215/EvalDeck.git
cd EvalDeck
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --config configs/example.yaml
```

打开 **http://127.0.0.1:7980**。示例配置包含演示用的任务组合，填写模型名称后即可预览 YAML。配置后端命令后，对应的评测按钮才会启用。

```bash
cp configs/example.yaml configs/local.yaml
# 填写服务脚本、任务组合和评测后端命令。
python app.py --config configs/local.yaml --check-config
python app.py --config configs/local.yaml --port 7980
```

命令参数、相对路径、YAML 格式和结果读取规则见[后端接入指南](docs/integration.md)。

## 代码结构

```text
EvalDeck/
├── app.py                  # 命令行入口
├── configs/example.yaml    # 部署配置模板
├── webui/
│   ├── config.py           # 配置加载与检查
│   ├── payloads.py         # 六类评测的 YAML 构造
│   ├── processes.py        # 进程生命周期与流式输出
│   ├── services.py         # 模型、攻击服务启动与停止
│   ├── evaluation.py       # 统一评测执行流程
│   ├── results.py          # 本次结果查找与预览
│   └── ui.py               # Gradio 组件与事件绑定
├── tests/test_workflow.py   # 配置与子进程回归测试
├── docs/integration.md     # 后端接入说明
└── requirements.txt
```

原来的 `test.py` 已替换为 `app.py` 和 `webui` 包，旧实现保留在 Git 历史中。本地部署配置与运行输出已加入 Git 忽略规则。

## 开发检查

```bash
python -m unittest discover -s tests -v
python app.py --config configs/example.yaml --check-config
```

默认仅监听本机。GPU 服务启动脚本面向 Linux，界面与 Python 子进程测试也可在 macOS 上运行。网络部署说明见[接入指南](docs/integration.md#development)。

## 许可证

[MIT](LICENSE)
