"""Build the uploadable A1 notebook, embedding the local runner verbatim."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def main():
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(True)})

    def code(source):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": dedent(source).strip().splitlines(True)})

    markdown("""
    # CS312 A1：Colab + Drive

    本 notebook 可单独上传到 Colab，不需要先 push 本地修改。它固定原仓库
    `d2407116240a41790fa0ff860b96dae3b08b72ec`，内嵌本地 runner，无需 Modal 或课程 checkpoint。

    **第一次先用 CPU runtime 完成 1–3；切换 GPU 后重新执行 1–2，再执行 4–9。**
    第 3 步的数据准备和第 8 步的训练默认关闭，按说明勾选后执行。
    Drive 存数据归档、checkpoint、指标与图；`/content` 存训练数据和临时 shuffle。
    完整数据约 19.7 GB，Drive 建议预留至少 30 GB，训练较多时继续检查剩余空间。

    日志使用真实 W&B offline 模式，无需 W&B 账号。额外保存 JSONL 曲线和默认 RMS 指标。
    notebook 使用 Colab 预装的 CUDA PyTorch，记录实际版本，不执行 `uv sync`；
    它可能与课程固定的 Torch 2.6 不同，应先完成 pilot 验证。

    [Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)说明了动态 GPU 可用性、
    订阅运行限制和 Drive I/O。此 notebook 已做静态与 runner 单元验证，尚未在真实 Colab GPU 上实跑。
    """)

    markdown("""
    ## 1. 挂载 Drive，取得固定版本的代码

    同一 runtime 只需运行一次。若更改下面的目录，请在导入仓库代码前完成。
    """)
    code(r'''
    import os
    import sys
    import json
    import subprocess
    from pathlib import Path
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/CS312")
    REPO = Path("/content/cs312-repo")
    REVISION = "d2407116240a41790fa0ff860b96dae3b08b72ec"
    DATASET = "dclm_9p6m_ctx1024"
    DRIVE_DATA = DRIVE_ROOT / "data" / DATASET
    LOCAL_DATA = Path("/content/cs312/data") / DATASET
    RELEASE_FILE = DRIVE_ROOT / "data_release.json"
    DATA_REVISION_FILE = DRIVE_ROOT / "data_revision.json"
    DATA_REPO = "kothasuhas/dl_alchemy_seq9p6m_context1024"
    MODEL_DIR = DRIVE_ROOT / "a1" / "ckpts"
    PLOT_DIR = DRIVE_ROOT / "a1" / "plots"
    for directory in (DRIVE_ROOT, MODEL_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not REPO.exists():
        subprocess.run(["git", "clone", "--quiet", "https://github.com/yuw136/CS312.git", str(REPO)], check=True)
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise RuntimeError("本地代码有改动，请先保存；本单元不会覆盖这些改动。")
        subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "--detach", REVISION], check=True)
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.environ["DL_ALCHEMY_DATA_DIR"] = str(LOCAL_DATA.parent)
    os.environ["DL_ALCHEMY_MODEL_DIR"] = str(MODEL_DIR)
    os.environ["DL_ALCHEMY_EPHEMERAL_DATA_DIR"] = "/content/cs312/tmp"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/content/cs312/wandb"
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    print("代码版本:", REVISION)
    print("Drive 数据:", DRIVE_DATA)
    print("本地数据:", LOCAL_DATA)
    print("持久 checkpoint:", MODEL_DIR)
    ''')

    markdown("""
    ## 2. 运行所需的小依赖

    保留预装 Torch。此步骤不安装 Modal，不下载数据，也不启动训练。
    """)
    code(r'''
    import importlib.util
    import importlib.metadata
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("当前 runtime 缺少 PyTorch，请选择带 PyTorch 的 Colab runtime。")
    packages = []
    for module, package in (("numpy", "numpy"), ("matplotlib", "matplotlib"), ("wandb", "wandb==0.31.0")):
        if importlib.util.find_spec(module) is None:
            packages.append(package)
    if packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)
    import torch
    print("Python:", sys.version.split()[0], "Torch:", torch.__version__)
    print("W&B:", importlib.metadata.version("wandb"))
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        print("当前 Python 超出原仓库声明的 3.10–3.12；本 notebook 使用预装环境，请先确认 pilot 正常。")
    ''')

    markdown("""
    ## 3. 一次性把公开数据下载到 Drive（建议 CPU runtime）

    第一次把 `PREPARE_DRIVE_DATA` 改为 True。此步骤在 Colab 中直接下载到 Drive，
    不经过你的电脑。先在 Drive 界面确认空间，Colab 计算订阅和 Drive 存储配额需分别确认。
    下载约 19.7 GB 后会校验大小与 SHA-256，最后写完成记录；中断后重跑本单元。
    下载器会复用已校验的完整文件，当前未完成的大文件会重新下载。

    成功后改回 False。以后每个 session 都跳过下载，只执行第 5 步复制。
    """)
    code(r'''
    #@title 一次性准备 Drive 数据
    PREPARE_DRIVE_DATA = False #@param {type:"boolean"}
    import hashlib
    import urllib.request
    if PREPARE_DRIVE_DATA:
        if RELEASE_FILE.exists():
            data_revision = json.loads(RELEASE_FILE.read_text())["revision"]
        elif DATA_REVISION_FILE.exists():
            data_revision = json.loads(DATA_REVISION_FILE.read_text())["revision"]
        else:
            api = f"https://huggingface.co/api/datasets/{DATA_REPO}/revision/main"
            with urllib.request.urlopen(api) as response:
                data_revision = json.load(response)["sha"]
            DATA_REVISION_FILE.write_text(json.dumps({"revision": data_revision}), encoding="utf-8")
        from download_data import download_default_data
        download_default_data(output_dir=DRIVE_DATA, repo_id=DATA_REPO, revision=data_revision)
        manifest_hash = hashlib.sha256((DRIVE_DATA / "manifest.json").read_bytes()).hexdigest()
        release = {"repo": DATA_REPO, "revision": data_revision, "manifest_sha256": manifest_hash}
        temporary = RELEASE_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(release, indent=2), encoding="utf-8")
        temporary.replace(RELEASE_FILE)
        print("数据已完整校验并保存在 Drive:", DRIVE_DATA)
    elif RELEASE_FILE.exists():
        print("已存在完成记录:", json.loads(RELEASE_FILE.read_text()))
    else:
        print("首次使用：检查 Drive 空间，将 PREPARE_DRIVE_DATA 设为 True 后运行本单元。")
    ''')

    markdown("""
    ## 4. 切换 GPU runtime 后检查硬件

    数据下载完成后，选择 GPU runtime（可用时 A100 或 L4），重新执行 1–2，跳过第 3 步下载。
    GPU 是否分配以及型号以此单元输出为准。每次新 runtime 都重新检查。
    """)
    code(r'''
    assert torch.cuda.is_available(), "请选择 GPU runtime 后重新运行第 1–2 步。"
    GPU_NAME = torch.cuda.get_device_name(0)
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    PRECISION = "mp" if native_bf16 else "fp32"
    print("GPU:", GPU_NAME)
    print("显存 GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    print("本次精度:", PRECISION, "原生 BF16:", native_bf16)
    print("可见 GPU 数:", torch.cuda.device_count())
    if torch.cuda.device_count() != 1:
        raise RuntimeError("此 A1 notebook 以单 GPU 为基线，请只暴露一张 GPU 后再运行。")
    ''')

    markdown("""
    ## 5. 每次新 runtime 把 Drive 数据复制到本地磁盘

    这一步约复制 20 GB。训练通过 memmap 读本地文件，临时 shuffle 也留在本地。
    已在当前 runtime 完整复制过时会跳过。Drive 中的原始数据不会被改动。
    """)
    code(r'''
    import shutil
    import hashlib
    if not RELEASE_FILE.exists():
        raise RuntimeError("还没有 Drive 数据完成记录，请先执行第 3 步。")
    release = json.loads(RELEASE_FILE.read_text())
    manifest_bytes = (DRIVE_DATA / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != release["manifest_sha256"]:
        raise RuntimeError("Drive manifest 与完成记录不符，请重新校验第 3 步。")
    manifest = json.loads(manifest_bytes)
    files = manifest["files"]
    receipt = LOCAL_DATA / "copied_release.json"
    local_ready = receipt.exists() and json.loads(receipt.read_text()) == release
    local_ready = local_ready and all(
        (LOCAL_DATA / entry["path"]).is_file()
        and (LOCAL_DATA / entry["path"]).stat().st_size == entry["size_bytes"]
        for entry in files
    )
    if local_ready:
        print("当前 runtime 已有完整本地副本。")
    else:
        total = sum(entry["size_bytes"] for entry in files)
        free = shutil.disk_usage("/content").free
        print("数据 GiB:", round(total / 1024**3, 2), "本地空闲 GiB:", round(free / 1024**3, 2))
        if free < total + 4 * 1024**3:
            raise RuntimeError("本地空间不足；需容纳数据、复制临时文件和 shuffle。请检查 /content 空间。")
        LOCAL_DATA.mkdir(parents=True, exist_ok=True)
        for entry in files:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("不支持的数据文件路径。")
            target = LOCAL_DATA / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".copying")
            digest = hashlib.sha256()
            print("复制并校验:", relative, flush=True)
            with (DRIVE_DATA / relative).open("rb") as source, temporary.open("wb") as destination:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
            if temporary.stat().st_size != entry["size_bytes"]:
                raise RuntimeError(f"文件大小不符: {relative}")
            if entry.get("sha256") and digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"文件校验不符: {relative}")
            temporary.replace(target)
        (LOCAL_DATA / "manifest.json").write_bytes(manifest_bytes)
        receipt.write_text(json.dumps(release, indent=2), encoding="utf-8")
        print("本地数据就绪:", LOCAL_DATA)
    os.environ["DL_ALCHEMY_DATA_REVISION"] = release["revision"]
    ''')

    markdown("""
    ## 6. 写入内嵌的本地 runner

    只需执行本单元。每条训练使用独立 Python 进程，保留原训练器和 RMS hook，
    W&B 使用 offline 模式，同时将原始指标写到 Drive 的 `history.jsonl`。
    同一 run 的配置发生变化时会要求换名字，防止误续跑。
    """)
    runner = (ROOT / "a1_runner.py").read_text(encoding="utf-8")
    code("#@title 写入本地 runner（无需编辑）\n"
         + "RUNNER_SOURCE = " + repr(runner) + "\n"
         + 'RUNNER = Path("/content/cs312_a1_runner.py")\n'
         + 'RUNNER.write_text(RUNNER_SOURCE, encoding="utf-8")\n'
         + 'print("runner:", RUNNER)')

    markdown("""
    ## 7. 选择 A1 实验

    首次保持 `MODE="pilot"`：6,400 条序列、100 个 optimizer updates。
    通过后改为 `MODE="p1_lr"`：每个 LR 使用 600,000 条序列，共 9,375 updates。
    `RUN_INDEX` 每次只选择一条 run，先跑索引 1 的 .003 基线，再跑 0 和 2 比较。

    `RUN_BATCH` 是本轮实验标识：同一次恢复保持不变；独立重复实验时换一个标识。
    `NUM_MICRO_BATCHES=16` 表示每次处理 4 条序列，有效 batch 仍是 64。
    若显存不足，可在新 run 开始前改成 32；已开始的 run 请保持配置用于恢复。
    原始 `p1_hyperparameters` 的 TODO 可以在本单元的配置里逐项实现，当前先提供 LR 对照。
    """)
    code(r'''
    #@title 选择一个实验，不会启动训练
    MODE = "pilot" #@param ["pilot", "p1_lr"]
    RUN_BATCH = "v1" #@param {type:"string"}
    RUN_INDEX = 0 #@param {type:"integer"}
    NUM_MICRO_BATCHES = 16 #@param [4, 8, 16, 32, 64] {type:"raw"}
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", RUN_BATCH):
        raise ValueError("RUN_BATCH 只使用英文字母、数字、点、下划线和连字符。")
    if MODE not in {"pilot", "p1_lr"}:
        raise ValueError("未知 MODE")
    count = 6_400 if MODE == "pilot" else 600_000
    rates = [.003] if MODE == "pilot" else [.001, .003, .01]
    JOBS = []
    for lr in rates:
        name = f"a1-{MODE}-{RUN_BATCH}-d8-lr{lr:g}-n{count}-{PRECISION}-mb{NUM_MICRO_BATCHES}"
        JOBS.append(dict(
            run_name=name, model_name="d8", learning_rate=lr,
            num_train_sequences=count, batch_size=64,
            num_micro_batches=NUM_MICRO_BATCHES, precision=PRECISION,
            num_evals=5 if MODE == "pilot" else 100,
            latest_checkpoint_frequency=25 if MODE == "pilot" else 100,
            data_seed=42, model_seed=42, resume_from_checkpoint=True,
            model_dir=str(MODEL_DIR), wandb_online=True,
            wandb_entity="local", wandb_project="cs312-a1",
            wandb_tags=["a1", "colab", MODE, RUN_BATCH],
        ))
    for index, job in enumerate(JOBS):
        print(index, job["run_name"])
    if not 0 <= RUN_INDEX < len(JOBS):
        raise ValueError("RUN_INDEX 超出配置列表范围。")
    print("本次选择:", JOBS[RUN_INDEX]["run_name"])
    print("精度:", PRECISION, "有效 batch: 64; microbatch:", 64 // NUM_MICRO_BATCHES)
    ''')

    markdown("""
    ## 8. 明确启动选中的一条训练

    将 `RUN_TRAINING` 勾为 True 后运行。保持本 cell 运行，查看 loss/ETA。
    停止按钮触发后请等待当前 update 结束并保存；runtime 强制回收时只能回到上次落盘的 checkpoint。
    断线后重新完成前面步骤，用相同配置重跑本单元，日志应出现 `Loading latest training checkpoint`。
    已完成的 run 会加载现有模型并跳过训练。不要递增 `stop_at_step` 作为通用分段方案。
    """)
    code(r'''
    #@title 启动选中的一条 run
    RUN_TRAINING = False #@param {type:"boolean"}
    import signal
    if not RUN_TRAINING:
        print("尚未启动。确认第 7 步配置后，勾选 RUN_TRAINING 再运行。")
    else:
        if not LOCAL_DATA.joinpath("copied_release.json").exists():
            raise RuntimeError("请先完成第 5 步复制数据。")
        job = JOBS[RUN_INDEX]
        config_dir = DRIVE_ROOT / "a1" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / (job["run_name"] + ".json")
        config_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", str(RUNNER), "--repo", str(REPO), "--config", str(config_path)]
        process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        interrupted = False
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            status = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            print("\n请求保存后停止，请等待当前 optimizer update 完成……", flush=True)
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for line in process.stdout:
                print(line, end="", flush=True)
            status = process.wait()
        finally:
            process.stdout.close()
        if status and not interrupted:
            raise RuntimeError(f"训练进程退出，状态码 {status}；查看上方具体错误。")
        print("输出目录:", MODEL_DIR / job["run_name"])
    ''')

    markdown("""
    ## 9. 查看曲线和结果

    此单元只读取当前 MODE/RUN_BATCH 的结果，按 optimizer step 合并日志。
    checkpoint 之后丢失的更新在实际恢复训练时会被清理，避免把旧尝试混入新曲线。
    保存 PNG 和 CSV 到 Drive；后续可以读取 `logging/rms/...` 做激活、参数、梯度分析。
    """)
    code(r'''
    import importlib.util
    import csv
    import math
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location("cs312_a1_runner", RUNNER)
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary = []
    for job in JOBS:
        run_dir = MODEL_DIR / job["run_name"]
        history = runner_module.read_history(run_dir / "history.jsonl")
        rows = list(history.values()) if isinstance(history, dict) else history
        rows = sorted(rows, key=lambda row: row["optimizer_step"])
        train_rows = [r for r in rows if isinstance(r.get("train_loss"), (int, float)) and math.isfinite(r["train_loss"])]
        val_rows = [r for r in rows if isinstance(r.get("val_loss"), (int, float)) and math.isfinite(r["val_loss"])]
        label = f"LR={job['learning_rate']:g}"
        if train_rows:
            axes[0].plot([r["optimizer_step"] + 1 for r in train_rows],
                         [r["train_loss"] for r in train_rows], label=label, alpha=.8)
        if val_rows:
            axes[1].plot([(r["optimizer_step"] + 1) * 64 * 1024 / 1e6 for r in val_rows],
                         [r["val_loss"] for r in val_rows], marker=".", label=label)
        summary.append(dict(run_name=job["run_name"], learning_rate=job["learning_rate"],
                            precision=job["precision"], num_train_sequences=job["num_train_sequences"],
                            completed=(run_dir / "model.pt").exists(),
                            last_logged_update=rows[-1]["optimizer_step"] + 1 if rows else "",
                            last_val_loss=val_rows[-1]["val_loss"] if val_rows else ""))
    for axis, title, x_label in zip(axes, ("Training loss", "Validation loss"),
                                    ("Optimizer updates", "Training positions (millions)")):
        axis.set(title=title, xlabel=x_label, ylabel="Loss")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    output = PLOT_DIR / f"{MODE}-{RUN_BATCH}.png"
    figure.savefig(output, dpi=160)
    plt.show()
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)
    print("图:", output, "\n表:", csv_path)
    ''')

    notebook = {"cells": cells, "metadata": {"colab": {"name": "A1_Colab.ipynb"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"cs312-a1-{index:02d}"
    destination = ROOT / "A1_Colab.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
