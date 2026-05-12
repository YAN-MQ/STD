# STD Rename And Full Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `scratch/06_realtime_emulation` to `scratch/STD`, update every project-local reproduction command and path reference to the new name, reset `issue.md`, then run the full `reproduce.md` workflow end-to-end while recording only newly encountered issues in `issue.md`.

**Architecture:** Treat this as a two-phase operation. Phase 1 is a source-of-truth migration: rename the directory, update build target names, absolute paths, wrapper scripts, and documentation so every official command points at `scratch/STD` and the rebuilt binaries live under `build/scratch/STD`. Phase 2 is a strict reproduction run driven by the updated `reproduce.md`; no opportunistic fixes are applied during reproduction, and any failure is logged into the freshly reset `issue.md` immediately with the exact command, symptom, log path, and blocking status.

**Tech Stack:** Bash, Markdown, CMake/ns-3 scratch targets, Python 3, PyTorch, libtorch, JSON/CSV artifacts

---

## Planned file structure and responsibilities

### Directory rename
- Rename: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation` → `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD`

### Source/build files that must change before reproduction
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/CMakeLists.txt`
  - Switch libtorch target/prefix/output references from `06_realtime_emulation` to `STD`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/run_all_window.sh`
  - Build `scratch_STD_realtime_satellite`; read binaries from `build/scratch/STD`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/OrbitShield_FL/config.py`
  - Update default `ns3_binary` path to `build/scratch/STD/...`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3.py`
  - Update `DEFAULT_NS3_BINARY`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_runtime.py`
  - Update `DEFAULT_NS3_BINARY`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_online.py`
  - Update `DEFAULT_NS3_BINARY`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_libtorch.py`
  - Update `PROJECT_ROOT`, `scratch_STD_federated_libtorch_runtime`, and `build/scratch/STD/...`

### Documentation/command files that must change before reproduction
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/README.md`
  - Replace every `scratch/06_realtime_emulation/...` path with `scratch/STD/...`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md`
  - Update every absolute path and CMake target to `STD`
  - Add explicit formal structured-compression commands so the document really covers the full run
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/federated_readme.md`
  - Update hard-coded `06_realtime_emulation` command paths to `STD`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`
  - Overwrite with a blank tracking template before reproduction starts

### Generated artifacts expected to be refreshed during reproduction
- Overwrite during rerun: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/dataset_cicids17/*`
- Overwrite during rerun: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/dataset_sti/*`
- Overwrite during rerun: formal experiment outputs under `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/**`
- Append during rerun: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

### Explicit assumption to validate with the user during review
- This plan updates **project-local source/docs/wrappers** immediately, then relies on rerunning the pipeline to refresh generated JSON/CSV summaries that still embed the old absolute path. It does **not** mass-edit historical generated artifacts by hand.

---

### Task 1: Inventory rename impact and establish guardrails

**Files:**
- Review: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/reproduce.md`
- Review: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/README.md`
- Review: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/issue.md`

- [ ] **Step 1: Capture repo status before any destructive rename**

Run:
```bash
git -C "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation" status --short
```
Expected: shows the exact pre-rename working tree state for the standalone repo.

- [ ] **Step 2: Capture all old-name references that must be resolved**

Run:
```bash
rg -n "06_realtime_emulation|scratch_06_realtime_emulation|/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation"
```
Expected: the current reference list includes at least `CMakeLists.txt`, `run_all_window.sh`, `README.md`, `reproduce.md`, `federated_readme.md`, `issue.md`, `4_train/OrbitShield_FL/config.py`, `4_train/scripts/train_federated_ns3.py`, `4_train/scripts/train_federated_ns3_runtime.py`, `4_train/scripts/train_federated_ns3_online.py`, and `4_train/scripts/train_federated_ns3_libtorch.py`.

- [ ] **Step 3: Confirm execution base for the rename**

Run:
```bash
ls "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch"
```
Expected: `06_realtime_emulation` is present and `STD` is absent before the rename.

---

### Task 2: Rename the directory and update build/runtime path constants

**Files:**
- Rename: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/CMakeLists.txt`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/run_all_window.sh`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/OrbitShield_FL/config.py`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3.py`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_runtime.py`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_online.py`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_libtorch.py`

- [ ] **Step 1: Rename the project directory from the parent path**

Run:
```bash
mv \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD"
```
Expected: the project is now located at `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD`.

- [ ] **Step 2: Update the libtorch CMake target/output naming**

Apply these exact replacements in `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/CMakeLists.txt`:
```cmake
EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/STD
```
```cmake
add_executable(
  scratch_STD_federated_libtorch_runtime
  federated_libtorch_runtime.cc
)
```
```cmake
target_compile_options(
  scratch_STD_federated_libtorch_runtime
```
```cmake
target_link_libraries(
  scratch_STD_federated_libtorch_runtime
```
```cmake
target_include_directories(
  scratch_STD_federated_libtorch_runtime
```
```cmake
set_target_properties(
  scratch_STD_federated_libtorch_runtime
```
```cmake
set_runtime_outputdirectory(
  "federated_libtorch_runtime"
  "${CMAKE_OUTPUT_DIRECTORY}/scratch/STD/"
  "scratch_STD_"
)
```

- [ ] **Step 3: Update the realtime wrapper to build and find the renamed scratch target**

Apply these exact replacements in `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/run_all_window.sh`:
```bash
cmake --build build --target scratch_STD_realtime_satellite -j"$(nproc)"
```
```bash
BIN=$(find "$NS3_DIR/build/scratch/STD" -maxdepth 1 -type f -name 'ns3.46.1-realtime_satellite-*' | head -n 1)
```

- [ ] **Step 4: Update default ns-3 binary paths in Python entry points**

Apply these exact replacements:

`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/OrbitShield_FL/config.py`
```python
ns3_binary: str = (
    "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/STD/"
    "ns3.46.1-federated_constellation-optimized"
)
```

`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3.py`
```python
DEFAULT_NS3_BINARY = (
    "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/STD/"
    "ns3.46.1-federated_constellation-optimized"
)
```

`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_runtime.py`
```python
DEFAULT_NS3_BINARY = (
    "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/STD/"
    "ns3.46.1-federated_constellation-optimized"
)
```

`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_online.py`
```python
DEFAULT_NS3_BINARY = (
    "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/STD/"
    "ns3.46.1-federated_constellation-optimized"
)
```

`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_libtorch.py`
```python
PROJECT_ROOT = Path("/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD")
```
```python
"scratch_STD_federated_libtorch_runtime",
```
```python
binary = BUILD_ROOT / "build" / "scratch" / "STD" / "ns3.46.1-federated_libtorch_runtime-optimized"
```

- [ ] **Step 5: Verify that source/build files no longer reference the old directory name**

Run:
```bash
rg -n "06_realtime_emulation|scratch_06_realtime_emulation" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/CMakeLists.txt" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/run_all_window.sh" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/OrbitShield_FL/config.py" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3.py" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_runtime.py" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_online.py" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/train_federated_ns3_libtorch.py"
```
Expected: no matches.

---

### Task 3: Update docs and reset `issue.md`

**Files:**
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/README.md`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/federated_readme.md`
- Modify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

- [ ] **Step 1: Replace every old absolute/project path in the docs**

Apply the following replacements everywhere they appear in `README.md`, `reproduce.md`, and `federated_readme.md`:
```text
scratch/06_realtime_emulation -> scratch/STD
/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation -> /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD
scratch_06_realtime_emulation_realtime_satellite -> scratch_STD_realtime_satellite
scratch_06_realtime_emulation_federated_constellation -> scratch_STD_federated_constellation
scratch_06_realtime_emulation_federated_libtorch_runtime -> scratch_STD_federated_libtorch_runtime
build/scratch/06_realtime_emulation -> build/scratch/STD
```

- [ ] **Step 2: Add explicit structured-compression reproduction commands to `reproduce.md`**

Insert these commands in the formal compression section of `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md`:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/run_structured_compression_candidates.py \
  --output_dir experiments/compression/structured_candidates_formal_tuned \
  --device cuda
/home/lithic/final/ns3-gpu-venv/bin/python scripts/select_structured_compression_winner.py \
  --candidates_dir experiments/compression/structured_candidates_formal_tuned
```
Expected prose change: the document should explicitly tell the runner how to reproduce the structured compression artifacts rather than only naming the summary file.

- [ ] **Step 3: Reset `issue.md` to a blank reproduction tracker**

Overwrite `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md` with:
```markdown
# issue

## 约束

- 严格按 `reproduce.md` 的正式流程执行复现实验。
- 复现阶段除“目录更名为 `STD` 与相关命令/路径更新”外，不做额外代码修复。
- 仅记录问题、命令、现象、日志位置与输出目录状态。
- 所有问题统一在全部复现结束并获得用户允许后再逐项修复。

## 复现总状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| 1. 目录更名与命令更新 | 未开始 | |
| 2. 环境准备与构建 | 未开始 | |
| 3. `cicids17` 数据集与单体模型 | 未开始 | |
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
```

- [ ] **Step 4: Verify that official docs and trackers no longer mention `06_realtime_emulation`**

Run:
```bash
rg -n "06_realtime_emulation|scratch_06_realtime_emulation" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/README.md" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/federated_readme.md" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md"
```
Expected: no matches.

---

### Task 4: Rebuild and smoke-test the renamed scratch targets

**Files:**
- Modify during build: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/**`
- Verify outputs: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/STD/*`

- [ ] **Step 1: Reconfigure CMake after the rename**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
cmake -S . -B build
```
Expected: configure completes; warnings are allowed but CMake generation succeeds.

- [ ] **Step 2: Build the renamed realtime and constellation scratch targets**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
cmake --build build --target scratch_STD_realtime_satellite -j"$(nproc)"
cmake --build build --target scratch_STD_federated_constellation -j"$(nproc)"
cmake --build build --target scratch_STD_federated_libtorch_runtime -j"$(nproc)"
```
Expected: all three targets finish with exit code `0`.

- [ ] **Step 3: Verify the renamed binaries exist where the updated wrappers expect them**

Run:
```bash
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build/scratch/STD" -maxdepth 1 -type f | sort
```
Expected: includes binaries matching:
```text
ns3.46.1-realtime_satellite-optimized
ns3.46.1-federated_constellation-optimized
ns3.46.1-federated_libtorch_runtime-optimized
```

- [ ] **Step 4: Verify the updated wrapper commands at least start correctly**

Run:
```bash
bash -n "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/run_all_window.sh"
bash -n "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/run_federated.sh"
bash -n "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/run_federated_ns3_online.sh"
bash -n "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts/run_federated_ns3_libtorch.sh"
```
Expected: no shell syntax errors.

---

### Task 5: Run the updated environment and build steps from `reproduce.md`

**Files:**
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`
- Verify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md`

- [ ] **Step 1: Install Python dependencies from the renamed project path**

Run:
```bash
source /home/lithic/final/ns3-gpu-venv/bin/activate
pip install -r /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/requirements.txt
python -c "import scapy"
```
Expected: `pip` succeeds and `import scapy` exits `0`.

- [ ] **Step 2: Execute the updated build commands exactly as documented**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
source /home/lithic/final/ns3-gpu-venv/bin/activate
cmake -S . -B build
cmake --build build --target scratch_STD_realtime_satellite -j"$(nproc)"
cmake --build build --target scratch_STD_federated_constellation -j"$(nproc)"
cmake --build build --target scratch_STD_federated_libtorch_runtime -j"$(nproc)"
```
Expected: all documented build commands complete successfully using the `STD` target names.

- [ ] **Step 3: If any command in this task fails, record the first fresh issue immediately**

Append an entry like this under `## 实时问题记录` in `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`:
```markdown
### Issue 1
- 阶段：2. 环境准备与构建
- 命令：`<paste exact failing command>`
- 现象：<what happened>
- 直接错误信息：`<stderr excerpt>`
- 相关输入/输出目录：<paths>
- 日志/证据位置：<log file path>
- 当前判断：<initial diagnosis only>
- 是否阻塞后续：是 / 否
- 备注：<extra context>
```
Expected: `issue.md` stays empty if nothing fails; otherwise the failure is recorded before moving on.

---

### Task 6: Reproduce `cicids17` dataset generation and the formal single model

**Files:**
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/fragments_window/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/captured_window/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/dataset_cicids17/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/checkpoints_gru_formal_tuned/cicids17_gru_best.pt`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

- [ ] **Step 1: Run the formal `cicids17` end-to-end pipeline from the renamed directory**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD
export PATH="/home/lithic/final/ns3-gpu-venv/bin:$PATH"
/home/lithic/final/ns3-gpu-venv/bin/python -c "import scapy"
MAX_PACKETS=50000 sudo -E bash ./run_all_window.sh
```
Expected: the script completes and refreshes `fragments_window/*`, `captured_window/*`, `dataset_cicids17/*`, and the formal `cicids17` checkpoint.

- [ ] **Step 2: Verify the expected output files exist after the run**

Run:
```bash
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/fragments_window" -maxdepth 1 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/captured_window" -maxdepth 1 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/dataset_cicids17" -maxdepth 1 -type f | sort
```
Expected: each directory contains the three formal `cicids17` outputs listed in `README.md`/`reproduce.md`.

- [ ] **Step 3: If the pipeline fails, log the exact failure in `issue.md` before retrying nothing**

Append a fresh issue entry under `## 实时问题记录` with:
```markdown
### Issue N
- 阶段：3. `cicids17` 数据集与单体模型
- 命令：`MAX_PACKETS=50000 sudo -E bash ./run_all_window.sh`
- 现象：<what stopped>
- 直接错误信息：`<stderr excerpt>`
- 相关输入/输出目录：`fragments_window`、`captured_window`、`dataset_cicids17`
- 日志/证据位置：<log file path>
- 当前判断：<initial diagnosis only>
- 是否阻塞后续：是 / 否
- 备注：<extra context>
```
Expected: no unlogged failure proceeds to the next stage.

---

### Task 7: Reproduce `STI` dataset preparation and the formal single model

**Files:**
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/dataset_sti/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/checkpoints_gru_formal_tuned/sti_gru_best.pt`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

- [ ] **Step 1: Regenerate the formal `STI` dataset**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD
/home/lithic/final/ns3-gpu-venv/bin/python 3_prepare_sti_dataset.py
```
Expected: rewrites `dataset_sti/train.npz`, `val.npz`, `test.npz`, and `metadata.json` with the new absolute project path rooted at `STD`.

- [ ] **Step 2: Reproduce the formal single-model `STI` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_gru.py \
  --dataset sti \
  --device cuda \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48 \
  --lr 0.0003 \
  --weight_decay 0.01 \
  --epochs 100 \
  --batch_size 128 \
  --early_stopping_patience 10 \
  --no-bidirectional \
  --output_dir checkpoints_gru_formal_tuned
```
Expected: refreshes `checkpoints_gru_formal_tuned/sti_gru_best.pt`.

- [ ] **Step 3: If either `STI` command fails, log it immediately and continue only if non-blocking**

Use this issue template:
```markdown
### Issue N
- 阶段：4. `STI` 数据集与单体模型
- 命令：`<paste exact failing STI command>`
- 现象：<what happened>
- 直接错误信息：`<stderr excerpt>`
- 相关输入/输出目录：`dataset_sti` 或 `4_train/checkpoints_gru_formal_tuned`
- 日志/证据位置：<log file path>
- 当前判断：<initial diagnosis only>
- 是否阻塞后续：是 / 否
- 备注：<extra context>
```

---

### Task 8: Reproduce formal comparison, ablation, and structured compression

**Files:**
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/comparison_formal_tuned/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/ablation_formal_tuned/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/compression/structured_candidates_formal_tuned/*`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

- [ ] **Step 1: Reproduce the formal model comparison run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/run_comparison.py \
  --output_dir experiments/comparison_formal_tuned \
  --include_traditional \
  --device cuda
```
Expected: refreshes `comparison_results.json` and `comparison_summary.csv` under `experiments/comparison_formal_tuned`.

- [ ] **Step 2: Reproduce the formal ablation run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
./run_ablation.sh \
  --output_dir experiments/ablation_formal_tuned \
  --comparison_config experiments/comparison_formal_tuned/comparison_results.json
```
Expected: refreshes `ablation_results.json` and `ablation_summary.csv` under `experiments/ablation_formal_tuned`.

- [ ] **Step 3: Reproduce the formal structured compression stage**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/run_structured_compression_candidates.py \
  --output_dir experiments/compression/structured_candidates_formal_tuned \
  --device cuda
/home/lithic/final/ns3-gpu-venv/bin/python scripts/select_structured_compression_winner.py \
  --candidates_dir experiments/compression/structured_candidates_formal_tuned
```
Expected: refreshes `structured_formal_summary.json` and candidate subdirectories.

- [ ] **Step 4: If any stage fails, log it under the matching issue section**

Use these exact stage labels in `issue.md`:
```markdown
- 阶段：5. 正式单机模型对比
- 阶段：6. 正式消融实验
- 阶段：7. 正式压缩 / 部署
```
Expected: each failure is isolated to the correct stage label and includes the failing command.

---

### Task 9: Reproduce Level 1 and Level 2 federated runs

**Files:**
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_formal_tuned/**`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_ns3_formal_tuned/**`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

- [ ] **Step 1: Reproduce the formal Level 1 `cicids17` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py \
  --dataset cicids17 \
  --device cuda \
  --output_dir experiments/OrbitShield_FL_formal_tuned/cicids17 \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_formal_tuned/cicids17/summary.json`.

- [ ] **Step 2: Reproduce the formal Level 1 `STI` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py \
  --dataset sti \
  --device cuda \
  --output_dir experiments/OrbitShield_FL_formal_tuned/sti \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48 \
  --full_eval
```
Expected: refreshes `experiments/OrbitShield_FL_formal_tuned/sti/summary.json`.

- [ ] **Step 3: Reproduce the formal Level 2 `cicids17` run using the updated path**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --trace_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3_formal_tuned/cicids17 \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_ns3_formal_tuned/cicids17/summary.json`.

- [ ] **Step 4: Reproduce the formal Level 2 `STI` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset sti \
  --trace_dir experiments/OrbitShield_FL_ns3/sti_trace \
  --output_dir experiments/OrbitShield_FL_ns3_formal_tuned/sti \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_ns3_formal_tuned/sti/summary.json`.

- [ ] **Step 5: If any federated run fails, log it under the correct stage label**

Use these exact labels in `issue.md`:
```markdown
- 阶段：8. `OrbitShield_FL` Level 1
- 阶段：9. `OrbitShield_FL + ns-3` Level 2
```

---

### Task 10: Reproduce Level 3 and Level 4B federated runs

**Files:**
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/**`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/**`
- Modify during reproduction: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`

- [ ] **Step 1: Reproduce the formal Level 3 `cicids17` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_online.py \
  --dataset cicids17 \
  --rounds 20 \
  --output_dir experiments/OrbitShield_FL_ns3_online_formal_tuned/cicids17 \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_ns3_online_formal_tuned/cicids17/summary.json`.

- [ ] **Step 2: Reproduce the formal Level 3 `STI` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_online.py \
  --dataset sti \
  --rounds 20 \
  --full_eval \
  --output_dir experiments/OrbitShield_FL_ns3_online_formal_tuned/sti \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_ns3_online_formal_tuned/sti/summary.json`.

- [ ] **Step 3: Reproduce the formal Level 4B `cicids17` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_libtorch.py \
  --dataset cicids17 \
  --rounds 20 \
  --local_epochs 1 \
  --batch_size 512 \
  --device cuda \
  --output_dir experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17 \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17/summary.json`.

- [ ] **Step 4: Reproduce the formal Level 4B `STI` run**

Run:
```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_libtorch.py \
  --dataset sti \
  --rounds 20 \
  --local_epochs 1 \
  --batch_size 512 \
  --device cuda \
  --output_dir experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/sti \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
Expected: refreshes `experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/sti/summary.json`.

- [ ] **Step 5: If any run fails, log it under the correct stage label**

Use these exact labels in `issue.md`:
```markdown
- 阶段：10. `OrbitShield_FL + ns-3 online` Level 3
- 阶段：11. `Level 4B: ns-3 + libtorch`
```

---

### Task 11: Final verification audit after the full reproduction pass

**Files:**
- Verify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md`
- Verify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md`
- Verify: `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/**`

- [ ] **Step 1: Verify the renamed doc and source commands contain no stale `06_realtime_emulation` references**

Run:
```bash
rg -n "06_realtime_emulation|scratch_06_realtime_emulation" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/README.md" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/reproduce.md" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/federated_readme.md" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/run_all_window.sh" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/CMakeLists.txt" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/OrbitShield_FL/config.py" \
  "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/scripts"
```
Expected: no matches.

- [ ] **Step 2: Verify the formal output files named in `README.md` exist after the rerun**

Run:
```bash
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/checkpoints_gru_formal_tuned" -maxdepth 1 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/comparison_formal_tuned" -maxdepth 1 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/ablation_formal_tuned" -maxdepth 1 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/compression/structured_candidates_formal_tuned" -maxdepth 2 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_formal_tuned" -maxdepth 2 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_ns3_formal_tuned" -maxdepth 2 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned" -maxdepth 2 -type f | sort
find "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned" -maxdepth 2 -type f | sort
```
Expected: every file called out in `README.md` exists under the renamed `STD` project root.

- [ ] **Step 3: Verify `issue.md` reflects only fresh problems from this run**

Run:
```bash
sed -n '1,220p' "/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD/issue.md"
```
Expected: either the tracker is still empty after the template, or it contains only issues encountered during this reproduction pass with updated `STD` commands.

---

## Self-review

### Spec coverage
- Rename project folder to `STD`: covered by Task 2.
- Update all reproduction commands to the new name: covered by Tasks 2–3 and verified again in Task 11.
- Clear `issue.md`: covered by Task 3.
- Record issues only when encountered: covered by Tasks 5–10.
- Reproduce the full process from `reproduce.md`: covered by Tasks 5–10.
- Provide a reviewable plan before execution: satisfied by this document.

### Placeholder scan
- No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
- Every code-changing task includes exact file paths, exact replacement snippets, or exact overwrite content.
- Every verification step includes an exact command and expected outcome.

### Type/path consistency
- Old path is always `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation`.
- New path is always `/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/STD`.
- Renamed target names consistently use `scratch_STD_*`.
- Binary output directory consistently uses `build/scratch/STD`.

Plan complete and saved to `docs/superpowers/plans/2026-05-08-std-rename-and-full-reproduction.md`. Please review it first; after approval we can choose either subagent-driven execution or inline execution.