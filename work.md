# 按 README 正式结构图逐项复核结果

## 1. 写作说明

- 本文档覆盖旧版 `work.md` 的全部内容。
- 本文档只保留根据 [README.md](./README.md) 中“最终正式实验链目录结构”自上而下逐项阅读得到的结果。
- 所有结论都对应当前目录中的实际文件与目录，不再保留历史阶段性叙述。

## 2. README 正式结构图逐项复核

README 中对应的正式结构从 [README.md](./README.md) 的“最终正式实验链目录结构”开始，当前逐项复核结果如下。

### 2.1 `fragments_window/`

README 标注：正式数据构建阶段的分片后 PCAP。

当前目录包含：

- [fragments_window/benign.pcap](./fragments_window/benign.pcap)
- [fragments_window/ddos.pcap](./fragments_window/ddos.pcap)
- [fragments_window/portscan.pcap](./fragments_window/portscan.pcap)

复核结论：

- 该目录对应 `cicids17` 三分类的正式回放输入。
- 三个文件分别对应 `benign / ddos / portscan` 三类流量。
- 这一层的作用是为后续 `ns-3` 回放提供正式输入 PCAP。

### 2.2 `captured_window/`

README 标注：`ns-3` 回放后的正式抓包输出。

当前目录包含：

- [captured_window/benign.pcap](./captured_window/benign.pcap)
- [captured_window/ddos.pcap](./captured_window/ddos.pcap)
- [captured_window/portscan.pcap](./captured_window/portscan.pcap)

复核结论：

- 该目录对应 `realtime_satellite.cc` 回放后的正式抓包结果。
- 三个文件分别是仿真输出侧的 `benign / ddos / portscan` 抓包结果。
- 这一层是后续特征提取脚本的直接输入。

### 2.3 `dataset_cicids17/`

README 标注：正式 `cicids17` 数据集。

当前正式文件为：

- [dataset_cicids17/train.npz](./dataset_cicids17/train.npz)
- [dataset_cicids17/val.npz](./dataset_cicids17/val.npz)
- [dataset_cicids17/test.npz](./dataset_cicids17/test.npz)

复核结论：

- 该目录是正式 `cicids17` 训练/验证/测试数据集。
- 结合当前项目代码与文档，这组数据对应 `(samples, 10, 18)` 的窗口特征输入。
- 标签为三分类：`benign = 0`、`ddos = 1`、`portscan = 2`。

### 2.4 `dataset_sti/`

README 标注：正式 `STI` 数据集。

当前正式文件为：

- [dataset_sti/metadata.json](./dataset_sti/metadata.json)
- [dataset_sti/train.npz](./dataset_sti/train.npz)
- [dataset_sti/val.npz](./dataset_sti/val.npz)
- [dataset_sti/test.npz](./dataset_sti/test.npz)

其中 [metadata.json](./dataset_sti/metadata.json) 复核结果如下：

- `dataset_name = STI`
- `input_shape = [1, 20]`
- `num_features = 20`
- `num_classes = 8`

标签映射为：

- `0 = Benign`
- `1 = Signal Disruption`
- `2 = UDP flood`
- `3 = Jamming`
- `4 = Bruteforce`
- `5 = Infiltration`
- `6 = DoS`
- `7 = DDoS`

全量规模为：

- `Train = (1273390, 1, 20)`
- `Val = (424461, 1, 20)`
- `Test = (424471, 1, 20)`

复核结论：

- `STI` 是当前正式保留的结构化表格数据集路线。
- 与 `cicids17` 的 PCAP→仿真→特征路径不同，`STI` 以 `(samples, 1, 20)` 形式适配当前训练框架。

### 2.5 `4_train/checkpoints_gru_formal_tuned/`

README 标注：本文模型正式单体 checkpoint。

当前正式文件为：

- [4_train/checkpoints_gru_formal_tuned/cicids17_gru_best.pt](./4_train/checkpoints_gru_formal_tuned/cicids17_gru_best.pt)
- [4_train/checkpoints_gru_formal_tuned/sti_gru_best.pt](./4_train/checkpoints_gru_formal_tuned/sti_gru_best.pt)

复核结论：

- 前者是 `cicids17` 的正式单体最优模型。
- 后者是 `STI` 的正式单体最优模型。
- 这两个 checkpoint 既支撑单机正式结果，也作为联邦训练 warm start 的正式初始化来源。

### 2.6 `4_train/experiments/comparison_formal_tuned/`

README 列出的正式文件为：

- [comparison_results.json](./4_train/experiments/comparison_formal_tuned/comparison_results.json)
- [comparison_summary.csv](./4_train/experiments/comparison_formal_tuned/comparison_summary.csv)

#### 2.6.1 `comparison_results.json`

复核结果：

- 保存了正式对比实验的完整配置、训练历史、评估指标与混淆矩阵。
- 当前正式主模型配置为：
  - `hidden_dim = 32`
  - `bidirectional = false`
  - `dropout = 0.4`
  - `conv_dim = 16`
  - `dsc_dim = 48`
  - `lr = 0.0003`
  - `weight_decay = 0.01`
  - `epochs = 100`
- 对比对象包括：
  - `dsc_cbam_gru`
  - `dsc_cbam_lstm`
  - `cnn_lstm`
  - `rf`
  - `id3`

#### 2.6.2 `comparison_summary.csv`

正式汇总结果如下：

| 模型 | family | params | flops | accuracy | f1 | composite_rank |
|---|---|---:|---:|---:|---:|---:|
| `dsc_cbam_gru` | deep | `12049` | `186334` | `0.990155` | `0.990162` | `1` |
| `cnn_lstm` | deep | `53611` | `296680` | `0.988997` | `0.989010` | `2` |
| `dsc_cbam_lstm` | deep | `48625` | `859698` | `0.986193` | `0.986217` | `3` |
| `rf` | traditional | `n/a` | `n/a` | `0.998476` | `0.998476` | `n/a` |
| `id3` | traditional | `n/a` | `n/a` | `0.993569` | `0.993567` | `n/a` |

复核结论：

- 当前正式深度学习模型对比中，`DSC-CBAM-GRU` 仍是综合排名第 `1` 的正式主模型。
- `RF` 与 `ID3` 保留传统模型精度结果，但不参与深度模型综合排序。

### 2.7 `4_train/experiments/ablation_formal_tuned/`

README 列出的正式文件为：

- [ablation_results.json](./4_train/experiments/ablation_formal_tuned/ablation_results.json)
- [ablation_summary.csv](./4_train/experiments/ablation_formal_tuned/ablation_summary.csv)

#### 2.7.1 `ablation_results.json`

复核结果：

- 保存了正式消融实验的完整训练历史、评估指标和混淆矩阵。
- 当前正式消融对象包括：
  - `dsc_cbam_gru`
  - `ablation_no_dsc`
  - `ablation_no_cbam`
  - `ablation_no_gru`

#### 2.7.2 `ablation_summary.csv`

正式汇总结果如下：

| 模型 | params | flops | accuracy | f1 | composite_rank |
|---|---:|---:|---:|---:|---:|
| `dsc_cbam_gru` | `12049` | `186334` | `0.990155` | `0.990162` | `1` |
| `ablation_no_dsc` | `13521` | `215614` | `0.987595` | `0.987606` | `2` |
| `ablation_no_cbam` | `11459` | `183584` | `0.980098` | `0.980083` | `3` |
| `ablation_no_gru` | `41169` | `104910` | `0.981378` | `0.981379` | `4` |

复核结论：

- 完整 `DSC-CBAM-GRU` 在正式消融实验中仍然排名第 `1`。
- 去除 `DSC / CBAM / GRU` 后，性能均有不同程度下降。

### 2.8 `4_train/experiments/compression/structured_candidates_formal_tuned/`

README 列出的正式文件为：

- [structured_formal_summary.json](./4_train/experiments/compression/structured_candidates_formal_tuned/structured_formal_summary.json)

复核结果：

- 该文件汇总了正式结构化压缩候选方案 `A1 / A2 / A3`。
- 正式压缩基线 `formal_tuned_baseline` 为：
  - `accuracy = 0.9892106065`
  - `f1 = 0.9892140364`
  - `parameter_count = 12049`
  - `latency_ms_per_sample_fp32 = 0.0056537917`
- 当前正式获胜候选为 `A2`：
  - `parameter_count = 4289`
  - `flops = 67626`
  - `int8_accuracy = 0.9882657726`
  - `int8_f1 = 0.9882672003`
  - `parameter_reduction_ratio = 0.6440368495`
  - `latency_reduction_ratio = 0.3288890301`
  - `meets_all_targets = true`

复核结论：

- 当前 README 中的正式“结构化压缩 + 动态 INT8”结果由该文件支撑。
- `A2` 是当前正式结构化压缩赢家。

### 2.9 `4_train/experiments/OrbitShield_FL_formal_tuned/`

README 结构图中展开为：

- [OrbitShield_FL_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_formal_tuned/cicids17)
- [OrbitShield_FL_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_formal_tuned/sti)

两者的 `summary.json` 复核结果如下：

| 数据集 | backend | 层级 | Test Accuracy | Test F1 |
|---|---|---|---:|---:|
| `cicids17` | `heuristic` | `Level 1` | `0.975495` | `0.975528` |
| `sti` | `heuristic` | `Level 1` | `0.997430` | `0.997431` |

复核结论：

- 该目录对应正式 Level 1 联邦实验结果。
- `heuristic` 表示联邦训练使用启发式动态拓扑后端。

### 2.10 `4_train/experiments/OrbitShield_FL_ns3_formal_tuned/`

README 结构图中展开为：

- [OrbitShield_FL_ns3_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_ns3_formal_tuned/cicids17)
- [OrbitShield_FL_ns3_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_ns3_formal_tuned/sti)

两者的 `summary.json` 复核结果如下：

| 数据集 | backend | 层级 | Test Accuracy | Test F1 |
|---|---|---|---:|---:|
| `cicids17` | `ns3` | `Level 2` | `0.968973` | `0.968898` |
| `sti` | `ns3` | `Level 2` | `0.997668` | `0.997669` |

复核结论：

- 该目录对应正式 Level 2 联邦实验结果。
- `ns3` 表示联邦训练使用离线 `ns-3 trace` 后端。

### 2.11 `4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/`

README 结构图中展开为：

- [OrbitShield_FL_ns3_online_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/cicids17)
- [OrbitShield_FL_ns3_online_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/sti)

两者的 `summary.json` 复核结果如下：

| 数据集 | backend | 层级 | Test Accuracy | Test F1 |
|---|---|---|---:|---:|
| `cicids17` | `ns3_online` | `Level 3` | `0.965651` | `0.965567` |
| `sti` | `ns3_online` | `Level 3` | `0.997293` | `0.997294` |

复核结论：

- 该目录对应正式 Level 3 联邦实验结果。
- `ns3_online` 表示每轮在线调用 `ns-3` 生成当前轮通信环境。

### 2.12 `4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/`

README 结构图中展开为：

- [OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17)
- [OrbitShield_FL_ns3_libtorch_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/sti)

两者的 `summary.json` 复核结果如下：

| 数据集 | backend | 层级 | Best Round | Test Accuracy | Test F1 |
|---|---|---|---:|---:|---:|
| `cicids17` | `ns3 + libtorch` | `Level 4B` | `7` | `0.971350` | `0.971306` |
| `sti` | `ns3 + libtorch` | `Level 4B` | `20` | `0.997116` | `0.997117` |

复核结论：

- 该目录对应正式 Level 4B 联邦实验结果。
- `ns3 + libtorch` 表示联邦训练运行时已经进入 C++ / `libtorch` 路线。

### 2.13 `4_train/scripts/`

README 标注：正式实验入口脚本。

当前脚本目录已检查，包含单体训练、对比、消融、压缩、导出、可视化以及 Level 1–Level 4B 联邦训练入口。

其中最关键的正式入口包括：

- [train_gru.py](./4_train/scripts/train_gru.py)
- [run_comparison.py](./4_train/scripts/run_comparison.py)
- [run_ablation.py](./4_train/scripts/run_ablation.py)
- [train_federated.py](./4_train/scripts/train_federated.py)
- [train_federated_ns3.py](./4_train/scripts/train_federated_ns3.py)
- [train_federated_ns3_online.py](./4_train/scripts/train_federated_ns3_online.py)
- [train_federated_ns3_libtorch.py](./4_train/scripts/train_federated_ns3_libtorch.py)

复核结论：

- `4_train/scripts/` 是当前正式实验链的统一入口目录。
- README 结构图虽然只画出 `scripts/` 目录本身，但它在实际工程中承载了单体、对比、消融、压缩和联邦各层级的正式入口。

### 2.14 `reproduce.md`

README 结构图最后列出的正式文件为：

- [reproduce.md](./reproduce.md)

复核结论：

- `reproduce.md` 是当前正式复现实验的统一操作入口说明。
- 其中包含：
  - 环境准备
  - 单体模型训练入口
  - 正式对比实验入口
  - 正式消融实验入口
  - 正式压缩入口
  - 正式 `Level 1–Level 4B` 联邦入口
- 如果只保留一个正式复现说明文件，应以 `reproduce.md` 为准。

## 3. README 结构图复核后的统一结论

经过这次按 README 正式结构图逐项复核，可以确认：

1. README 中保留的正式目录与当前项目中的正式数据、正式 checkpoint、正式结果汇总和正式入口脚本是一致的。
2. [comparison_formal_tuned](./4_train/experiments/comparison_formal_tuned)、[ablation_formal_tuned](./4_train/experiments/ablation_formal_tuned)、[structured_candidates_formal_tuned](./4_train/experiments/compression/structured_candidates_formal_tuned) 和各级 [OrbitShield_FL_*_formal_tuned](./4_train/experiments) 目录已经形成一套可直接对应论文结果表的正式实验归档。
3. README 中“最终正式实验链目录结构”不是示意图，而是可以被当前目录下实际文件逐项落地验证的正式交付结构。
4. 当前 `work.md` 仅保留这套基于 README 正式结构图的复核结果，不再混用历史阶段性记录。
