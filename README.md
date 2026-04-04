# BirdCLEF Pipeline

这是一个从 BirdCLEF notebook 拆出来的本地脚本工程，保留了原始 notebook 的主要流程：

- EDA
- PCEN 特征提取与缓存
- 模型训练
- TTA 推理与提交文件生成

项目使用 `uv` 管理依赖，核心代码已经按比赛工程的方式拆分到 `src/birdclef/` 下。

## 目录说明

- `birdclef_pipeline.py`：轻量启动入口
- `src/birdclef/config.py`：配置与命令行参数
- `src/birdclef/dataset.py`：数据表构建、Dataset、Sampler
- `src/birdclef/audio.py`：音频读取、PCEN、缓存预计算
- `src/birdclef/model.py`：模型定义
- `src/birdclef/train.py`：训练与验证
- `src/birdclef/infer.py`：推理与提交文件生成
- `src/birdclef/eda.py`：基础 EDA 与可视化

## 环境准备

```bash
uv sync
```

## 常用命令

先看帮助：

```bash
uv run birdclef --help
```

运行 EDA：

```bash
uv run birdclef eda
```

预计算 PCEN 缓存：

```bash
uv run birdclef cache --cache-workers 4
```

训练指定 fold：

```bash
uv run birdclef train --folds 0,1 --epochs 3
```

运行推理并生成提交文件：

```bash
uv run birdclef infer --tta-crops 3
```

也可以直接调用入口脚本：

```bash
uv run python birdclef_pipeline.py train --folds 0
```

## 数据路径

默认读取的是 notebook 里的 Kaggle 风格路径：

```text
./kaggle/input/birdclef-2026/
```

如果你在本地或 Windows 上运行，建议显式传入数据根目录：

```bash
uv run birdclef train --data-root D:/birdclef-2026 --folds 0
```
