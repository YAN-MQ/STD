# 面向低轨卫星网络的轻量化星上AI威胁预测模型

本科毕业设计项目，围绕低轨卫星网络中的短窗口威胁预测任务，完成了从数据构建、单体模型训练、联邦学习到部署评估的正式实验链路。

## 项目说明

当前仓库中的正式实验口径统一为：

- 本文模型：单向 GRU 版 `DSC-CBAM-GRU`
- 正式压缩 / 部署口径：结构化压缩模型（结构化压缩 + 短程微调 + `fbgemm` 动态INT8）
- 联邦学习主模型：仍使用未压缩本文模型

如果需要复现实验，请直接以 `scratch/06_realtime_emulation/reproduce.md` 为准。为便于快速核对，当前论文采用的正式结果口径概括如下。

## 正式结果总览

### 单机主模型结果

| 数据集 | Accuracy | F1 | 参数量 | FLOPs |
|---|---:|---:|---:|---:|
| `cicids17` | 99.02% | 99.02% | 12.0K | 186.3K |
| `STI` | 99.82% | 99.82% | 12.4K | 25.6K |

### 深度学习模型对比（`cicids17`）

| 模型 | Accuracy | F1 | 参数量 | FLOPs |
|---|---:|---:|---:|---:|
| DSC-CBAM-GRU（本文） | 99.02% | 99.02% | 12.0K | 186.3K |
| CNN-LSTM | 98.90% | 98.90% | 53.6K | 296.7K |
| DSC-CBAM-LSTM | 98.62% | 98.62% | 48.6K | 859.7K |

### 消融实验（`cicids17`）

| 模型配置 | Accuracy | F1 | 参数量 | FLOPs |
|---|---:|---:|---:|---:|
| DSC-CBAM-GRU | 99.02% | 99.02% | 12.0K | 186.3K |
| 去除DSC | 98.76% | 98.76% | 13.5K | 215.6K |
| 去除GRU | 98.14% | 98.14% | 41.2K | 104.9K |
| 去除CBAM | 98.01% | 98.01% | 11.5K | 183.6K |

### 正式压缩 / 部署结果

| 模型 | Accuracy | F1 | 参数量变化 | 单样本CPU时延 |
|---|---:|---:|---:|---:|
| 原始FP32模型 | 98.92% | 98.92% | 基线 | 0.005654 ms |
| 结构化压缩+动态INT8模型 | 98.83% | 98.83% | -64.40% | 0.003794 ms |

### 联邦学习层级结果

#### `cicids17`

| 层级 | Accuracy | F1 |
|---|---:|---:|
| Level 1 | 97.55% | 97.55% |
| Level 2 | 96.90% | 96.89% |
| Level 3 | 96.57% | 96.56% |
| Level 4B | 97.14% | 97.13% |

#### `STI`

| 层级 | Accuracy | F1 |
|---|---:|---:|
| Level 1 | 99.74% | 99.74% |
| Level 2 | 99.77% | 99.77% |
| Level 3 | 99.73% | 99.73% |
| Level 4B | 99.71% | 99.71% |

## 最终正式实验链目录结构

```text
scratch/06_realtime_emulation/
├── fragments_window/                              # 正式数据构建阶段的分片后 PCAP
├── captured_window/                               # ns-3 回放后的正式抓包输出
├── dataset_cicids17/                              # 正式 cicids17 数据集
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── dataset_sti/                                   # 正式 STI 数据集
│   ├── metadata.json
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── 4_train/
│   ├── checkpoints_gru_formal_tuned/              # 本文模型正式单体 checkpoint
│   │   ├── cicids17_gru_best.pt
│   │   └── sti_gru_best.pt
│   ├── experiments/
│   │   ├── comparison_formal_tuned/
│   │   │   ├── comparison_results.json
│   │   │   └── comparison_summary.csv
│   │   ├── ablation_formal_tuned/
│   │   │   ├── ablation_results.json
│   │   │   └── ablation_summary.csv
│   │   ├── compression/
│   │   │   └── structured_candidates_formal_tuned/
│   │   │       └── structured_formal_summary.json
│   │   ├── OrbitShield_FL_formal_tuned/
│   │   │   ├── cicids17/
│   │   │   └── sti/
│   │   ├── OrbitShield_FL_ns3_formal_tuned/
│   │   │   ├── cicids17/
│   │   │   └── sti/
│   │   ├── OrbitShield_FL_ns3_online_formal_tuned/
│   │   │   ├── cicids17/
│   │   │   └── sti/
│   │   └── OrbitShield_FL_ns3_libtorch_formal_tuned/
│   │       ├── cicids17/
│   │       └── sti/
│   └── scripts/                                   # 正式实验入口脚本
└── reproduce.md                                   # 最终正式复现实验入口说明
```

## 正式复现入口

请直接参考：
- `scratch/06_realtime_emulation/reproduce.md`

其中包含：
- 本文模型训练入口
- 正式单机模型对比入口
- 正式消融入口
- 正式压缩入口
- 正式 Level 1–Level 4B 联邦入口

## 说明

- 本 README 只保留最终正式实验链的项目说明、精简正式结果表与目录结构；
- 论文正文与 `reproduce.md` 为正式结果与复现命令的权威来源；
- 历史实验目录与中间结果如果仍保留在 `experiments/` 下，不应视为当前论文正式结果。
