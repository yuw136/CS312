# 在 Colab 做 Assignment 1

使用同目录的 **`A1_Colab.ipynb`**。把这个 notebook 文件单独上传到 Colab 即可；它会拉取 `yuw136/CS312` 的固定版本 `d240711`，并写入内嵌的 A1 runner，不需要先把新增文件推送到 GitHub。

推荐顺序是：**pilot 短跑 → 一个默认学习率基线 → 三个学习率对照**。A1 不依赖课程提供的 checkpoint，也不需要课程 Modal workspace。它使用仓库现有的本地 CUDA trainer；原始入口和配置方式见 [非 Modal 指南](../gpu/README.md) 与 [A1 P1 starter](../experiments/a1_basics/p1_hyperparameters.py)。

## 第一次使用

先用 CPU runtime 完成数据准备，再切 GPU，避免下载占用 GPU 时间。以下编号对应 notebook 的操作顺序。

1. **挂载 Drive、拉取固定版本。** 运行第 1 节，完成 Drive 授权，确认输出的仓库版本和保存路径。
2. **加载最少依赖。** 运行第 2 节。它沿用 Colab 自带的 CUDA PyTorch，不执行仓库的 `uv sync`。
3. **把数据准备到 Drive，只做一次。** 将 `PREPARE_DRIVE_DATA = True`，运行第 3 节。下载约 **19.7 GB** 的完整原始 token 数据；仓库 downloader 会检查文件大小与 SHA-256，完成后在 Drive 的 `data_release.json` 中记录固定的 Hugging Face 数据版本。成功后把开关改回 `False`。不要因为 pilot 只用少量序列，就预计下载量也会相应减小。
4. **切换 GPU runtime。** 在 Colab 的“更改运行时类型”中选择 GPU；订阅账号如果能选 A100，优先 A100，否则可选 L4。重新连接后重跑第 1、2 节；第 3 节保持 `False`，使用 Drive 上已经验证的数据。GPU 型号与可用性由 Colab 实际分配决定。
5. **复制训练数据到本机盘。** 运行第 5 节，将 Drive 数据复制到 `/content`。训练从本机盘读 token，checkpoint 和日志存回 Drive。runtime 更换后通常需要再次复制。
6. **写入 A1 runner。** 运行第 6 节。runner 已内嵌在 notebook 中，无需另行上传 Python 文件。
7. **选择一个实验。** 运行配置单元格，先设 `MODE = "pilot"`、`RUN_INDEX = 0`，并为这批实验设置清晰、唯一的 `RUN_BATCH`。
8. **明确启动选中的任务。** 第 8 节的 `RUN_TRAINING` 默认是 `False`；勾选为 `True` 后运行启动单元格。它只运行 `RUN_INDEX` 指向的一个任务，每次使用新的 Python 进程；切换索引后，需要重新运行配置和启动单元格。默认开关避免“全部运行”直接开始训练。
9. **查看曲线和汇总。** 训练结束后运行绘图单元格。根目录是 `MyDrive/CS312`；checkpoint 和每个任务的日志位于 `a1/ckpts/<run_name>`，图和汇总位于 `a1/plots`。以单元格打印的完整路径为准。

Drive 需要容纳约 20 GB 原始数据以及训练输出，建议开始前至少有 **30 GB 空间，40 GB 更宽裕**。Colab 订阅不自动增加 Drive 存储配额，请检查自己的 Drive 剩余容量。Colab 官方也建议减少运行中对 Drive 的小文件读写；这里因此把训练数据复制到 `/content`。[Colab FAQ](https://research.google.com/colaboratory/faq.html)

## 从 pilot 到 A1 P1

| 配置 | 用途 | 训练预算 |
|---|---|---|
| `MODE = "pilot"` | 检查数据、训练、保存和绘图是否贯通 | 6,400 条序列，100 次 optimizer update |
| `MODE = "p1_lr"` | 比较学习率，其余条件保持一致 | 每个配置 600,000 条序列；默认 LR 为 `.001`、`.003`、`.01` |

pilot 的曲线只用于检查流程。完成后切换到 `p1_lr`，先从任务列表中选择 `.003` 的索引做基线，再做其他两个学习率。默认列表有三个任务；使用 `RUN_INDEX = 0`、`1`、`2` 分别选择，并逐个启动。以单元格打印的任务列表为准。

默认有效 batch 为 **64**，`num_micro_batches = 16`，即每次处理 4 条序列并累积 16 次梯度，再更新一次参数。若显存不足，优先增加可整除 64 的 microbatch 数，并使用新 run 名，保持有效 batch 和训练预算。runner 会根据 GPU 是否原生支持 BF16 自动选择 BF16 混合精度或 FP32，不假定订阅账号一定获得某个 GPU 或精度能力。

每批实验共享 `RUN_BATCH`，每个配置有独立名字和保存目录。改变实验设计、进行独立重复或重新开始时使用新的 `RUN_BATCH`；要续跑同一个实验时保留原值。

不要直接启动 [A1 P2 starter](../experiments/a1_basics/p2_scaling_law_reliability.py)：其原始配置会生成 **24 次完整训练**。先完成 P1，确认单次耗时和资源余量后，再设计自己的小规模对照。

## 断线、日志和续跑

runtime 可能在额度用完、闲置或达到生命周期上限时结束；订阅也不保证无限连续运行，具体限制会变化。[Colab FAQ](https://research.google.com/colaboratory/faq.html)

恢复时重跑 Drive 挂载、仓库与依赖单元格，把已有 Drive 数据重新复制到 `/content`，再写入 runner。使用相同的 `MODE`、`RUN_BATCH`、`RUN_INDEX` 和训练配置启动，日志中应出现 `Loading latest checkpoint` 一类的加载提示；恢复位置是最后成功保存的 checkpoint，最后一段尚未保存的计算需要重做。

不要通过不断增大 `stop_at_step` 来续跑：保持原始训练配置和学习率调度周期，让 checkpoint 恢复继续执行。改变配置可能形成不同任务，或者改变实验含义。

runner 使用真实 W&B 的离线模式，并保存本地 JSONL 指标与 RMS 日志，无需 W&B 账号或登录。普通训练和附加 logger 的结果可在 Drive 中查看，绘图单元格直接读取保存的指标。它不会向课程 Modal 提交任务。

## 验证范围

这份 notebook 面向 A1 的单 GPU 实验。pilot 缩短了训练预算，其数值不能当作课程完整实验结论。完整的 `p1_lr` 保留默认的序列预算与有效 batch，但不同 GPU、精度和执行后端可能带来数值或耗时差异。

交付前仅做了 notebook 结构、代码静态检查和 helper 的标准库测试，**尚未在真实 Colab GPU 上完成端到端训练**。首次使用应先完成 pilot，再开始完整训练。
