# CS312 使用说明与 Colab 迁移方案

核对日期：2026-11-04。代码版本：`d2407116240a41790fa0ff860b96dae3b08b72ec`，分支 `main`。本文省略软件安装、账号注册与环境配置步骤，只介绍作业工作流和迁移设计。

## 1. 仓库怎么用

### 作业入口

这是一套小型自回归语言模型实验框架。核心训练接口是 `train(TrainConfig(...))`；Modal 负责远程提交和存储，并不是训练算法的一部分。

| 作业 | 目录 | 主要内容 |
| --- | --- | --- |
| A1 | `experiments/a1_basics/` | 超参数、scaling law、实验变异、放大效应、loss curve 等 starter |
| A3 | `experiments/a3_optimization/` | mode connectivity、LR schedule、稳定性边缘、Hessian |
| A4 | `experiments/a4_stability/` | Transformer/RNN 深度、残差混合、循环更新 |
| A5 | `experiments/a5_capacity/` | 深度宽度、readout、MoE、looping |

当前快照没有 A2 目录，A1 的 `p6_activation_gradient_norms.py` 是空文件。仓库中有 starter 和 TODO，课程 handout 单独分发；不能从目录推断出完整题目要求。

依据：[主 README](../README.md)、[A3](../experiments/a3_optimization/README.md)、[A4](../experiments/a4_stability/README.md)、[A5](../experiments/a5_capacity/README.md)。

### 如何“锁定一个 assignment”

仓库没有 `--assignment 3` 这样的统一开关，也没有 assignment tag；当前只有 `main`。通常同时固定两个东西：代码提交与要运行的题目模块。

例如，在本次核对的代码上做 A3，可以执行：

```bash
git switch -c work/a3 d240711
git rev-parse HEAD
```

然后只修改、运行 `experiments/a3_optimization/` 中选定题目的入口。分支用于保存自己的工作；提交 SHA 用于复现。这里给出的是建议命令，本文没有替你创建工作分支。

建议以当前版本为起点，它包含后续修复。只有 handout 明确要求旧版本时才回到旧发布点：A3 `57c60bb`，A4 `4350166`，A5 `0efbc3d`。以后同步上游时，可以在保存自己的改动后切回 `main` 执行 `git pull --ff-only`，再有意识地把更新合入工作分支。

### 一次实验的日常流程

1. 阅读所选 problem 文件和对应 handout，确认要比较的变量。
2. 修改文件中的 `RUNS`、`TrainConfig` 或底部 `launch_one(...)`。先选一条配置或一对对照，再扩大 sweep。
3. 给实验使用独立的 `run_name` / `run_name_suffix` 和 W&B tag；实验方案变化时换名字，避免读到已有模型或混入旧结果。
4. 从仓库根目录用 `uv run python -m 包.模块` 执行。Modal launcher 通常提交 detached job，命令退出只说明已提交；查看打印的 Modal 链接、W&B 和产物确认完成。
5. 训练结束后再运行依赖 checkpoint 的测量或画图阶段。

默认配方是 d8、600,000 条序列、context 1024、有效 batch 64、AdamW、LR .003、linear decay、seed 42；约 614.4M 输入位置、9,375 个 optimizer updates。`experiments.smoke.modal_smoke_train` 也是这个完整配方，并非十几步的快速测试。

训练参数定义见 [train.py](../train.py)。Modal 产物保存在 `/root/data/ckpts/<run_name>/`；本地入口写入 `model_dir/<run_name>/`。默认会恢复 `latest.pt`；完整训练结束后还有 `config.json`、`model.pt`、`metadata.json`。复用已完成的 run 名称通常会直接加载现有模型。

### 常用实验命令

下面是原仓库默认执行路径，通常会访问 Modal。Colab 的替代调用见下一节。

```bash
# A1 P1：先编辑 RUNS
uv run python -m experiments.a1_basics.p1_hyperparameters

# A3 P1：每个阶段完成后，再执行下一条
uv run python -m experiments.a3_optimization.p1_mode_connectivity prefix
uv run python -m experiments.a3_optimization.p1_mode_connectivity branches
uv run python -m experiments.a3_optimization.p1_mode_connectivity measure

# A4 P1：先检查文件底部 launch_one 调用
uv run python -m experiments.a4_stability.p1_transformer_depth

# A5 P3：先检查底部选择的 MoE variant
uv run python -m experiments.a5_capacity.p3_moe
```

如果只是想熟悉“定义实验 → 训练 → 画图”，已有完整模板：

```bash
uv run python -m experiments.lr_tuning.launch_lr_tuning
# 等训练完成、W&B 指标可用后
uv run python -m experiments.lr_tuning.plot_lr_tuning
```

它默认提交六个 LR，不是一次训练。`EXPERIMENT_KEY` 同时控制 tag 和画图筛选；修改实验时同步更改这个 key。输出图为 `experiments/lr_tuning/plots/lr-tuning-v1_loss_summary.png`。A1 P2 的默认列表更大，包含 24 条训练配置；运行任何 starter 前都应先看配置列表。

A4/A5 可以先只打印配置：

```python
from experiments.a4_stability.sweeps import launch_one

launch_one("rnn", .001, dry_run=True)
# 确认配置后，选用本地 GPU 路径
launch_one("rnn", .001, local=True)
```

A5 的 `experiments.a5_capacity.sweeps` 也有同名接口。某些题目模块一运行就会提交任务；`dry_run=True` 需要在 Python 调用中明确指定，不能假设所有模块默认 dry-run。

## 2. Modal 权限的实际边界

Modal 本身面向普通用户开放；注册会创建自己的 workspace。课程 README 中的 Stanford 邮箱、课程邀请、GPU 限额属于课程 workspace 的规则。[Modal 官方 Workspace 文档](https://modal.com/docs/guide/workspaces)

这个 fork 已经提供 [README_MODAL_OWN_WORKSPACE.md](../README_MODAL_OWN_WORKSPACE.md)，明确支持没有课程 workspace 的独立学习者。你可以使用自己的 Modal 资源和公开训练数据；课程提供的共享 checkpoint 则另有访问条件。个人 workspace 的计算和存储由自己的账户承担，不能使用课程额度。

Colab 仍然是可行选择：直接使用相同的 PyTorch 训练代码和模型，迁移远程执行、数据路径及持久化方式即可。[现有非 Modal GPU 说明](../gpu/README.md)

## 3. 推荐的 Colab 迁移结构

| 原来的职责 | Colab 对应方案 |
| --- | --- |
| `modal_train.launch_training_jobs(configs)` | 在当前 runtime 顺序执行配置；普通训练调用 `train(cfg)` |
| A5 专用训练 hook | 保留 `experiments.a5_capacity.sweeps.train(cfg)` 包装器 |
| Modal 共享数据 Volume | 公开 token 数据放在 `/content/cs312/data/` 等本地磁盘目录 |
| Modal 用户 checkpoint Volume | `model_dir` 指向已挂载的 Drive 持久目录 |
| 多个独立 H100 任务并发 | 先按实际分配的一张 GPU 串行运行，各个 run 可独立续跑 |
| Modal Secret 注入 W&B key | 使用普通 W&B 的本机/notebook 认证；训练不需要 Modal secret |
| Modal fetch 测量结果 | 本地直接写 JSON/CSV/PNG，并保存到 Drive |

Colab 的 GPU 型号、可用额度和超时是动态的，免费 runtime 最长 12 小时且可能更早结束；VM 空闲或达到生命周期后会被删除，因此不能只依赖 `/content` 保存训练状态。[Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)

### 数据：保留配方与缩小实验要分清

公开数据源为 `kothasuhas/dl_alchemy_seq9p6m_context1024`。现有 `download_data` 会下载完整约 19.7 GB 的原始数据及 metadata，并校验大小和 SHA-256。减小 `num_train_sequences` 不会让下载器只下载相应的部分。

建议首先复用原始数据格式和下载器，把数据放在 runtime 本地磁盘。Drive 保存一份长期副本时，每次训练前复制到本地，避免 memmap 的随机访问直接打到 Drive；官方也建议减少 Drive 的频繁小文件 I/O。[Colab FAQ 的 Drive I/O 说明](https://research.google.com/colaboratory/faq.html)

下载归档应保留用于清单和校验的 `manifest.json`；训练读取的每个 split 必须包含 `metadata.json` 和 `tokens.bin`。默认 `data_seed=42` 会从全量数据的全局排列中取前缀，并在临时磁盘 materialize，600k 行 uint16 token 约另需 1.2 GB。memmap 不会把全量 20 GB 一次载入内存。

为省空间另做小数据集也可行，但必须正确生成 metadata 并明确标成缩小实验。直接裁剪原文件前几行、再使用 seed42，不等价于原来的全局 seed42 子集；`data_seed=None` 也改变数据顺序。用于最终比较时应记录这些差异。依据：[data.py](../data.py)、[download_data.py](../download_data.py)。

### 精度、显存与 batch

现有 `precision="mp"` 使用 BF16 autocast，不是 FP16；当前支持的值只有 `mp`、`bf16`、`fp32`。运行时用 `torch.cuda.is_bf16_supported(including_emulation=False)` 判断原生 BF16 支持。T4/Turing 没有原生 BF16 Tensor Core，首版建议 `fp32`；原生支持 BF16 的 GPU 可沿用 `mp`。不要仅根据 GPU 名称或默认允许模拟的检测结果决定精度。[NVIDIA Turing 文档](https://docs.nvidia.com/cuda/turing-tuning-guide/)、[仓库所固定 PyTorch 2.6 的官方实现](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/__init__.py)

这里 `batch_size` 是有效 batch。保持 `batch_size=64`，把 `num_micro_batches` 提高到 16，则每个 microbatch 是 4 条序列，累计 16 次才做一次 optimizer update；验证也按这个较小 batch 运行。这样可以保留有效 batch 和 LR schedule 的 update 数，但浮点累加与部分模型的 batch 相关行为仍可能有差异。16 只是试跑起点，不保证所有模型都能在分配到的 GPU 上容纳。

`num_micro_batches>1` 自动关闭主 trainer 的 `torch.compile`，适合先验证链路。不过 A4 RNN runtime 和 A5 MoE 内部仍有自己的编译/CUDA graph 路径；如遇兼容问题，需要分别加 eager 开关。当前 `torch_compile_mode` 只是 mode，并不是一个总开关。

如以后确实需要 FP16，应增加独立 precision 模式及相应 `GradScaler` 路径；不要只把现有 BF16 dtype 换成 FP16。

### 一段首轮试跑代码

以下是迁移后的调用示例，假设 notebook 已能访问仓库、数据和 Drive，不包含安装或认证步骤。路径设置必须早于首次导入仓库模块，因为默认配置在 import 时求值。重复修改这些默认路径时应重启 runtime，或显式传入 dataset/config 路径。

```python
import os

os.environ["DL_ALCHEMY_DATA_DIR"] = "/content/cs312/data"
os.environ["DL_ALCHEMY_EPHEMERAL_DATA_DIR"] = "/content/cs312/tmp"
os.environ["DL_ALCHEMY_MODEL_DIR"] = "/content/drive/MyDrive/CS312/ckpts"

import torch
from train import TrainConfig, train

assert torch.cuda.is_available(), "当前 runtime 没有可用 CUDA GPU"
precision = (
    "mp" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp32"
)

cfg = TrainConfig(
    run_name="colab-pilot-d8-v1",
    num_train_sequences=6_400,  # 缩小试跑：一轮 100 个 optimizer updates
    batch_size=64,
    num_micro_batches=16,      # 每次实际处理 4 条序列
    precision=precision,
    num_evals=5,
    latest_checkpoint_frequency=25,
    resume_from_checkpoint=True,
    wandb_online=False,
)
train(cfg)
```

先确认 loss、显存、每步耗时和 checkpoint 写入，再用新 run 名扩大训练预算。`wandb_online=False` 仅适合第一轮链路验证：当前代码会同时关闭 metric logger，因此这段试跑不会得到完整的 RMS/sharpness 等作业指标。正式实验需要保留 W&B 在线日志，或者实现有等价指标的 JSONL/CSV sink。

A4 的自定义配置可以 `cfg = sweeps.config("d8", .003, ...)` 后调用原生 `train(cfg)`。A5 应使用自己的包装器，保留 quantile routing 的 optimizer hook 和中途 sharing switch：

```python
from experiments.a5_capacity import sweeps as a5

cfg = a5.config(
    "cplm-r64", .003,
    run_name="colab-a5-cplm-r64-v1",
    num_train_sequences=6_400,
    num_micro_batches=16,
    precision=precision,
    wandb_online=False,
    latest_checkpoint_frequency=25,
)
a5.train(cfg)
```

上面的 A5 例子也只是 pilot。A5 中途共享切换实验应保留其专用约束，不能统一套用这套覆盖参数。

### 持久化与恢复

- `model_dir` 放到 Drive，沿用原有 `latest.pt`、`step_N.pt`、最终模型机制。checkpoint 频率先试 25～100 updates，再按写入耗时调整。
- 同一次中断恢复保留相同 run 名、代码、数据、精度、microbatch 与训练配置。断线重连后调用同一配置，恢复最近一次成功写入的 checkpoint；强制回收 VM 时，最后一段未持久化进度可能丢失。
- 每条 run 额外记录 Git SHA、GPU、精度、有效 batch、microbatch 数、数据 revision、seed、训练序列数和耗时。修改方案时换 run 名，避免把不同实验当作续跑。
- 先测完整 checkpoint 写入时间，必要时改成“本地完成写入，再将完整文件同步到 Drive”，并明确同步成功才算持久化。
- 不要把反复增大 `stop_at_step` 当成通用分段续跑：它参与命名，而且达到该停止点也可能产生完整模型，重跑可能被跳过。长训练保留完整目标，靠 checkpoint 恢复；需要 fork 的实验沿用题目的专用接口。

依据：[checkpointing.py](../checkpointing.py)、[train.py](../train.py)。

## 4. 不同作业的迁移难点

| 范围 | 建议做法 | 需要单独处理的内容 |
| --- | --- | --- |
| A1 普通训练 | 配置交给本地 `train(cfg)`，小规模串行扫描 | 多数 starter 顶层导入 Modal launcher；抽取配置或把远程 import 延迟到远程分支 |
| A3 P1/P2 | prefix、branches、measure 仍按依赖顺序本地运行 | P1 的 volume 检查、测量远程包装需替换；P2 默认 launcher 也是 Modal |
| A3 P3 sharpness | 保留测量定义，先缩减采样频率/实验格点 | `probe="batch"` 要求 `num_micro_batches=1`，不能直接套梯度累积；日志目前依赖 `wandb_online=True` |
| A3 P4 Hessian | 用本地 checkpoint 调底层测量函数，按单任务试算成本 | `--local` 主要覆盖测量；模块仍有顶层 Modal import/decorator。训练阶段不自动由 `--local` 改成本地 |
| A4 | 使用 `sweeps.config` / `launch_one(..., local=True)` | 课程 P1/P2 参考权重、RNN 内部编译与 CUDA graph、FP32 probes 的额外显存 |
| A5 | 使用 `sweeps.train` / `launch_one(..., local=True)` | MoE 内部编译；中途 sharing switch 应保留默认四个 microbatch、单 GPU、无 EMA 等专用约束，并匹配 switch step |

Hessian 的二阶计算、完整 RNN probes 和大 grid 应放在普通训练跑通之后；减小 probe 批量、Lanczos 步数或迭代次数会改变估计质量，需要在报告中注明，不能直接声称复现原参考值。

### 课程资源不会随执行平台自动迁移

| 资源 | 当前仓库给出的外部可用性 | Colab 处理方式 |
| --- | --- | --- |
| 原始训练/验证 token 数据 | 有公开 Hugging Face 下载源 | 直接复用 downloader 和数据格式 |
| A3 P4 的 AdamW/SGDM 参考 checkpoint | 位于课程共享 Volume；个人 workspace 文档明确不包含这些权重 | 向课程方获取副本，或训练并使用自己的 run；报告中注明来源变化 |
| A4 P1/P2 的 105 个参考 outcomes | W&B 项目公开；manifest 与权重在课程 Volume | 曲线分析可用公开项目；权重诊断需要副本或自训 |
| A5 P1 depth/width 参考结果 | W&B 曲线公开；manifest/history 文件与权重在课程 Volume | 优先拿 manifest 或导出公开指标；本地 loader 需要 `A5_REFERENCE_ROOT` 或显式 `root` |

A5 P1 默认画图命令也会启动 Modal CPU worker 去取 manifest。换到 Colab 后，要有本地 manifest 再走 `--local`，或另写读取公开 W&B 指标的分析入口。公开曲线不代表已拥有权重和原始 manifest。

资源说明：[A3 README](../experiments/a3_optimization/README.md)、[A4 README](../experiments/a4_stability/README.md)、[A5 README](../experiments/a5_capacity/README.md)。

## 5. 建议的实施顺序与验收

1. 新增一个 Colab notebook，固定代码 SHA；提供 assignment/problem、variant、pilot/full、run 名和输出路径的选择。先支持普通 d8 和 A4 单条配置。
2. 实现串行本地 runner，复用 `TrainConfig` 和现有模型；把相关 launcher 的 Modal import 放进远程分支。A5 走自己的训练包装器。
3. 跑约 100 updates 的 pilot，确认有限 loss、验证结果、最终模型和持久 checkpoint；另起一条未完成 run 验证中断恢复确实接着 optimizer step 训练。
4. 以实际测得的每步耗时估算 9,375 updates 与验证、存盘的总耗时，再决定单次训练规模。先做 2～3 个 LR 的小 grid，每次只改变一个变量。
5. 单独接入 A3 checkpoint 测量、A4/A5 参考数据和诊断。拿不到课程权重的部分明确使用自训替代，不阻碍先做其他题目。
6. 最后才考虑 FP16、编译开关和更大的搜索。对改动后的模块运行已有相应 CPU checks，再用真实 Colab GPU 验证数值和恢复行为。

本次交付是基于源码与官方资料的使用说明和迁移方案，没有启动 Modal/Colab 训练、下载约 20 GB 数据、修改训练器或创建 notebook。实际显存、速度和完整作业的可运行规模需要在分配到的 Colab GPU 上测量。
