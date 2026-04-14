# BirdCLEF 2026 Pipeline

本地化的 BirdCLEF 2026 算法工程流水线，提供基于 `uv` 管理的完整端到端方案。

## 1. 环境与数据准备

建议在 Python 3.11+ 环境下运行：

```bash
# 安装依赖
uv sync

# 下载官方数据集（需配置好 ~/.kaggle/kaggle.json）
mkdir -p kaggle/input/birdclef-2026
cd kaggle/input/birdclef-2026
uvx kaggle competitions download -c birdclef-2026
unzip -q birdclef-2026.zip
rm birdclef-2026.zip
cd ../../../
```

## 2. 快速起步

项目支持拆分运行，也可一键串行运行。

### 运行数据分析 (EDA)
统计数据分布并生成可视化图像：
```bash
uv run birdclef eda
```

### 预计算音频缓存
将音频提取为 PCEN 特征缓存，极大加速训练过程：
```bash
uv run birdclef cache --cache-workers 8
```

### 本地模型训练
支持 K-Fold 训练，默认自动保存最优的验证集模型。
```bash
uv run birdclef train --folds 0,1,2,3,4 --epochs 6 --batch-size 32
```

### 离线推理并导出提交文件
读取本地 `test_soundscapes` 并利用你训练的权重生成打分结果：
```bash
uv run birdclef infer --tta-crops 3
```

### 一键执行全流程
依次自动完成 EDA、缓存计算、训练和推理：
```bash
uv run birdclef all --cache-workers 8 --epochs 6 --folds 0,1,2,3,4
```
