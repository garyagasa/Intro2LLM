# Chapter 2 Lab 2: BERT 预训练项目

这是一个从零实现的 BERT 预训练示例项目，已经整理为分层清晰的工程结构，而不是单文件脚本。它包含数据准备、WordPiece 词元器训练、MLM 预训练、以及训练后的 fill-mask 推理演示。

## 项目结构

```text
.
├── main.py
├── requirements.txt
├── requirements-cu128.txt
├── data/
│   ├── train.txt
│   └── test.txt
├── artifacts/
│   ├── tokenizer/
│   └── model/
└── src/
	└── bert_pretrain/
		├── config.py
		├── data.py
		├── inference.py
		├── model.py
		├── preprocess.py
		├── tokenizer.py
		└── trainer.py
```

模块职责很直接：`data.py` 负责拉取并保存原始文本，`tokenizer.py` 负责训练 WordPiece，`preprocess.py` 负责把文本变成可训练样本，`model.py` 构建 BERT MLM，`trainer.py` 负责训练参数、保存和 wandb 接入，`inference.py` 负责训练后的示例推理。

## 运行环境

目标平台是 H100，CUDA 版本是 12.8。建议直接使用 PyTorch 的 cu128 轮子，并在训练时启用 bf16，这也是 H100 上最合适的默认精度。

### 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 会继续引用 `requirements-cu128.txt`，其中已经包含 PyTorch 的 cu128 索引源。如果你的机器已经预装了可用的 GPU 版 PyTorch，也可以保留现有环境，只补装其余依赖。

如果要启用 wandb 记录，先执行：

```bash
wandb login
```

## 运行方式

直接执行入口脚本即可跑完整流程：

```bash
python main.py
```

默认流程会按下面顺序执行：

1. 下载并合并 `bookcorpus` 与 `wikipedia`。
2. 切分训练集和测试集，并写入 `data/train.txt` 和 `data/test.txt`。
3. 基于训练集训练 WordPiece tokenizer，并保存到 `artifacts/tokenizer/`。
4. 将文本转成 MLM 训练样本，并进行 BERT 预训练。
5. 训练结束后自动运行一个小的 fill-mask 演示。

## 训练输出

训练过程中会生成这些目录：

- `data/`：原始文本切分后的纯文本文件。
- `artifacts/tokenizer/`：词表和 tokenizer 配置。
- `artifacts/model/`：最终模型权重与配置。
- `artifacts/model/checkpoints/`：训练中的 checkpoint。
- wandb 会记录训练损失、评估损失和超参数，项目名默认是 `bert-pretrain`。

## H100 建议配置

- 默认会启用 `bf16` 和 `tf32`，如果检测不到 CUDA，会自动关闭这两项。
- 如果显存仍然紧张，优先调小 `per_device_train_batch_size`，其次调小 `gradient_accumulation_steps`。
- 这个示例用的是 BERT-base 级别的配置，真正大规模预训练时可以继续扩展 `model.py` 里的结构参数。

## 说明

- `HF_ENDPOINT` 默认设置为 `https://hf-mirror.com`，适合国内下载环境。
- 当前实现已经修正了原始单文件脚本里分词后未清理原始文本列的问题，避免后续分块时把原文列一起混入。
- 如果你想把它继续扩展成命令行工具，可以直接在 `main.py` 之上再加一层参数解析，不会影响现有模块边界。