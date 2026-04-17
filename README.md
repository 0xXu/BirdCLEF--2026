# BirdCLEF 2026 Pipeline


## 核心设计

- 输入：32 kHz mono waveform cache，训练时动态裁 10s，在线生成 `256x256` LogMel。
- 验证：默认 multi-label stratified group split，按 `audio_id` 分组防泄漏。
- 训练：EfficientNet + GeM，mixup/cutmix，masked BCE/Focal，rare-class 与 taxonomy-aware sampler。
- 半监督：teacher ensemble -> soundscape pseudo labels -> student training。
- 后处理：temporal smoothing、soundscape-level max boost、per-class calibration。
- 产物：checkpoint、OOF、AUC 报表、calibration、hard negatives、pseudo labels、submission。

## 环境与数据

安装依赖：

```bash
uv sync
```

比赛数据目录：

```text
kaggle/input/birdclef-2026/
```

本地 backbone 权重目录：

```text
kaggle/input/models/timm/tf-efficientnet/pytorch/tf-efficientnet-es/1/
```

示例：

```text
kaggle/input/models/timm/tf-efficientnet/pytorch/tf-efficientnet-es/1/model.safetensors
```

训练不会联网下载 timm 权重，缺失本地权重会直接报错。

## 1. 构建 Teacher Cache

```bash
uv run birdclef cache \
  --output-dir kaggle/working_teacher_v1 \
  --cache-workers 4
```

cache 保存的是 waveform，不是固定谱图；训练阶段会动态裁片并生成 LogMel。

## 2. 训练 Teacher

```bash
uv run birdclef train \
  --output-dir kaggle/working_teacher_v1 \
  --folds 0,1,2,3,4 \
  --epochs 6 \
  --batch-size 8 \
  --cache-workers 4 \
  --num-workers 0 \
  --rare-class-threshold 30 \
  --hard-negative-threshold 0.35
```

主要输出：

```text
model_fold*.pth
fold_metrics.csv
oof_predictions*.csv
per_class_auc.csv
per_class_auc_rank.csv
source_auc.csv
taxonomy_group_auc.csv
calibration.csv
hard_negatives.csv
training_history.png
```

## 3. 生成 Pseudo Labels

```bash
uv run birdclef pseudo \
  --output-dir kaggle/working_teacher_v1 \
  --pseudo-path kaggle/working_teacher_v1/pseudo_labels.csv \
  --tta-crops 3 \
  --pseudo-batch-size 16 \
  --pseudo-min-primary-prob 0.50 \
  --pseudo-label-prob 0.35 \
  --pseudo-mask-prob 0.10 \
  --calibration-path kaggle/working_teacher_v1/calibration.csv \
  --soundscape-smooth-kernel 0.1,0.2,0.4,0.2,0.1 \
  --soundscape-max-boost 0.12 \
  --soundscape-boost-threshold 0.20
```

pseudo 生成逻辑：

- checkpoint 按 fold `val_auc` 加权 ensemble
- 每个 5s frame 使用 TTA crops
- 应用 temporal smoothing 与 soundscape-level max boost
- 如存在 `calibration.csv`，先做 per-class calibration
- 输出完整 `pseudo_<class>` soft targets

## 4. 构建 Student Cache

```bash
uv run birdclef cache \
  --output-dir kaggle/working_student_v1 \
  --pseudo-path kaggle/working_teacher_v1/pseudo_labels.csv \
  --cache-workers 4
```

## 5. 训练 Student

```bash
uv run birdclef train \
  --output-dir kaggle/working_student_v1 \
  --pseudo-path kaggle/working_teacher_v1/pseudo_labels.csv \
  --hard-negative-path kaggle/working_teacher_v1/hard_negatives.csv \
  --calibration-path kaggle/working_teacher_v1/calibration.csv \
  --folds 0,1,2,3,4 \
  --epochs 6 \
  --batch-size 8 \
  --cache-workers 4 \
  --num-workers 0 \
  --pseudo-sampling-weight 0.40 \
  --rare-class-threshold 30 \
  --hard-negative-threshold 0.35
```

Student 会生成自己的 `calibration.csv` 和 `hard_negatives.csv`，最终推理优先使用 student 产物。

## 6. Inference

本地 inference 用于验证格式和流程。最终 leaderboard 得分必须在 Kaggle hidden test 环境中生成。

```bash
uv run birdclef infer \
  --output-dir kaggle/working_student_v1 \
  --tta-crops 3 \
  --calibration-path kaggle/working_student_v1/calibration.csv \
  --soundscape-smooth-kernel 0.1,0.2,0.4,0.2,0.1 \
  --soundscape-max-boost 0.12 \
  --soundscape-boost-threshold 0.20
```

输出：

```text
kaggle/working_student_v1/submission.csv
```

## CV 与报表

默认 CV：

```bash
uv run birdclef train \
  --cv-strategy mlsgkf_audio_id \
  --group-col audio_id \
  --folds 0,1,2,3,4
```

补充验证：

```bash
uv run birdclef train --cv-strategy soundscape_site --folds 0
uv run birdclef train --cv-strategy soundscape_date --folds 0
uv run birdclef train --cv-strategy source_holdout --folds 0
```

合并分批训练得到的 OOF：

```bash
uv run birdclef merge-oof --output-dir kaggle/working_teacher_v1
```

## 目录约定

```text
kaggle/input/birdclef-2026/                      # 比赛数据
kaggle/input/models/timm/.../model.safetensors   # 本地 backbone 权重
kaggle/working_teacher_v1/                       # teacher 输出
kaggle/working_student_v1/                       # student 输出
```

## 默认建议

- Apple Silicon：使用 `--num-workers 0`，`--batch-size 8`；内存紧张时降到 `4`。
- Cache：优先 `--cache-workers 4`，磁盘和 CPU 都有余量再提高。
- Pseudo 阈值：默认 `0.50 / 0.35`；伪标签过少时降到 `0.40 / 0.30`。
- 最终提交：使用 student checkpoint 和 student `calibration.csv`。
