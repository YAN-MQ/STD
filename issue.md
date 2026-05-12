# issue

## 约束

- 严格按 `reproduce.md` 的正式流程执行复现实验。
- 复现阶段除“目录更名为 `STD` 与相关命令/路径更新”外，不做额外代码修复。
- 仅记录问题、命令、现象、日志位置与输出目录状态。
- 所有问题统一在全部复现结束并获得用户允许后再逐项修复。

## 复现总状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| 1. 目录更名与命令更新 | 已完成 | 已更名为 `STD`，相关路径、target 与文档命令已更新 |
| 2. 环境准备与构建 | 已完成 | `requirements.txt` 解析问题已修复，`scapy` 已安装，三个 `scratch_STD_*` target 构建成功 |
| 3. `cicids17` 数据集与单体模型 | 已完成 | 已完成全流程复现；期间发现 `run_all_window.sh` 曾调用旧训练入口，已修复并重跑成功 |
| 4. `STI` 数据集与单体模型 | 未开始 | |
| 5. 正式单机模型对比 | 未开始 | |
| 6. 正式消融实验 | 未开始 | |
| 7. 正式压缩 / 部署 | 未开始 | |
| 8. `OrbitShield_FL` Level 1 | 未开始 | |
| 9. `OrbitShield_FL + ns-3` Level 2 | 未开始 | |
| 10. `OrbitShield_FL + ns-3 online` Level 3 | 未开始 | |
| 11. `Level 4B: ns-3 + libtorch` | 未开始 | |

## 问题记录模板

### Issue N
- 阶段：
- 命令：
- 现象：
- 直接错误信息：
- 相关输入/输出目录：
- 日志/证据位置：
- 当前判断：
- 是否阻塞后续：是 / 否
- 备注：

## 实时问题记录

### Issue 1
- 阶段：2. 环境准备与构建
- 命令：`source /home/lithic/final/ns3-gpu-venv/bin/activate && pip install -r /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/requirements.txt && python -c "import scapy"`
- 现象：按更新后的 `reproduce.md` 安装 Python 依赖时，`pip` 在解析 `requirements.txt` 第 3 行前直接失败，依赖没有安装成功；随后单独执行 `python -c "import scapy"` 也失败。
- 直接错误信息：`ERROR: Invalid requirement: 'Requirements for training DSC-CBAM-LSTM models.': Expected semicolon (after name with no version specifier) or end`；`ModuleNotFoundError: No module named 'scapy'`
- 相关输入/输出目录：`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/requirements.txt`
- 日志/证据位置：`/tmp/std_repro_stage2_pip_install.log`
- 当前判断：`requirements.txt` 含有不符合 pip 语法的裸文本说明行，导致正式依赖安装命令无法执行；当前 `scapy` 尚未安装，因此会阻塞后续 `cicids17` 全流程复现。
- 是否阻塞后续：是
- 备注：已修复为注释行，随后重新执行安装成功，`scapy 2.7.0` 可正常导入，三个 `scratch_STD_*` target 已构建完成。

### Issue 2
- 阶段：3. `cicids17` 数据集与单体模型
- 命令：`cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD && export PATH="/home/lithic/final/ns3-gpu-venv/bin:$PATH" && /home/lithic/final/ns3-gpu-venv/bin/python -c "import scapy" && MAX_PACKETS=50000 sudo -E bash ./run_all_window.sh`
- 现象：正式 `cicids17` 全流程在首次执行时进入 `sudo` 前失败，未实际启动 `run_all_window.sh` 的 root 相关步骤。
- 直接错误信息：`sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper`；`sudo: a password is required`
- 相关输入/输出目录：输入 `/home/lithic/final/data`；目标输出 `fragments_window`、`captured_window`、`dataset_cicids17`
- 日志/证据位置：`/tmp/claude-1000/-home-lithic-final/8b925587-c680-4f5a-87cf-d1d48e478103/tasks/bz8d3ktlt.output`
- 当前判断：当前复现路线本身依赖 `sudo` 创建 TAP 接口、抓包和回放；问题不在脚本逻辑，而在本会话尚未完成可复用的 `sudo` 认证。
- 是否阻塞后续：是
- 备注：已在后续重试中使用用户提供的 `sudo` 密码完成认证并成功启动全流程。

### Issue 3
- 阶段：3. `cicids17` 数据集与单体模型
- 命令：`cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD && export PATH="/home/lithic/final/ns3-gpu-venv/bin:$PATH" && /home/lithic/final/ns3-gpu-venv/bin/python -c "import scapy" && MAX_PACKETS=50000 sudo -E bash ./run_all_window.sh`
- 现象：命令本身执行成功，分片、回放、抓包和特征提取都完成，但脚本最后训练出的并不是 `reproduce.md` 声明的“本文正式单体模型”输出路径与配置。
- 直接错误信息：无（流程 exit code 0）
- 相关输入/输出目录：`fragments_window`、`captured_window`、`dataset_cicids17`、`4_train/checkpoints_gru`、`4_train/checkpoints_gru_formal_tuned`
- 日志/证据位置：`/tmp/claude-1000/-home-lithic-final/8b925587-c680-4f5a-87cf-d1d48e478103/tasks/bdvhqqgbu.output`
- 当前判断：根因是 `run_all_window.sh` 内部最后一步仍调用 `python3 scripts/train_gru.py --data_dir ../dataset_cicids17 --epochs 20 --num_classes 3 --input_dim 18`，实际输出到了 `4_train/checkpoints_gru/cicids17_gru_best.pt`，且训练参数是旧口径；这与 `reproduce.md` 中“正式单体模型应落到 checkpoints_gru_formal_tuned/cicids17_gru_best.pt，且采用 formal tuned 配置”的要求不一致。
- 是否阻塞后续：是
- 备注：下一步应优先最小修复 `run_all_window.sh` 的最后训练入口，使其走 `run_train.sh` 或等价的 formal tuned 训练命令，然后重新执行 `cicids17` 全流程。
