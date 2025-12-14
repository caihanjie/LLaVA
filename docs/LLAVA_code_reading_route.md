# LLaVA 代码阅读与调试路线图

> 按“从入口到细节、从主链路到分支”的阅读顺序设计，所有结论均可回指到具体文件/函数。

## 目录
- [1. 全局地图](#1-全局地图)
- [2. 最小推理闭环（先跑起来再细看）](#2-最小推理闭环先跑起来再细看)
- [3. 模型结构拆解](#3-模型结构拆解)
- [4. 训练链路](#4-训练链路)
- [5. 推理与 Serving 链路](#5-推理与-serving-链路)
- [6. 评测链路](#6-评测链路)
- [7. 调试断点与观察项](#7-调试断点与观察项)
- [8. 自测清单](#8-自测清单)

---

## 1. 全局地图

### 1.1 目录结构（2~3 层，含职责）
- 根目录：`README.md`、`pyproject.toml`、`scripts/`（训练/预训练脚本）、`docs/`（使用文档）、`images/`（示例图）、`llava-v1.5-7b/`（示例权重）。
- `llava/`：核心代码
  - `constants.py`：特殊 token、索引常量。
  - `conversation.py`：对话模板定义与 prompt 生成。
  - `mm_utils.py`：图像预处理、特殊 token 插入、StoppingCriteria。
  - `model/`
    - `builder.py`：`load_pretrained_model` 入口，加载 tokenizer/LLM/vision/projector。
    - `llava_arch.py`：多模态组装、视觉特征插入、`prepare_inputs_labels_for_multimodal`。
    - `language_model/`：`llava_llama.py`、`llava_mistral.py`、`llava_mpt.py`，定义 LLaVA 版 CausalLM。
    - `multimodal_encoder/`：CLIP 视觉塔 `CLIPVisionTower`。
    - `multimodal_projector/`：视觉到文本的 projector（linear/MLP）。
  - `serve/`：推理 CLI、Gradio/Web、分布式 worker。
  - `train/`：训练入口、数据集/Collator、Trainer 扩展。
  - `eval/`：评测脚本集合（VQA、ScienceQA、MMBench 等）。
- `playground/`：示例数据。
- `scripts/`：训练/预训练/LoRA/QLoRA shell，Deepspeed 配置（`zero*.json`）。

### 1.2 关键入口清单
- 推理：`llava/serve/cli.py`（最小 CLI）、`llava/serve/model_worker.py`（worker）、`llava/serve/gradio_web_server.py`（Gradio）、`llava/eval/run_llava.py`（单轮问答评测）。
- 训练：`llava/train/train.py`（主入口），`train_mem.py`（FlashAttention2），`train_xformers.py`（xformers），对应 shell 在 `scripts/`.
- 模型定义：`llava/model/builder.py`（加载）、`llava/model/llava_arch.py`（融合逻辑）、`llava/model/multimodal_encoder/*`（视觉塔）、`llava/model/multimodal_projector/*`（对齐层）。
- 数据处理：`llava/train/train.py` 中 `LazySupervisedDataset`/`DataCollatorForSupervisedDataset`；推理预处理在 `llava/mm_utils.py`、`conversation.py`。
- 评测：`llava/eval/run_llava.py`（基准）、其它任务专用脚本在 `llava/eval/*.py`。

### 1.3 端到端张量流概览（文字版）
1. 输入图像 → `mm_utils.process_images`：`PIL.Image` 预处理（pad/anyres/default）→ `pixel_values` `[B,3,H,W]`（默认 336x336，float32/float16）。
2. 视觉编码器 `CLIPVisionTower.__call__`（`llava/model/multimodal_encoder/clip_encoder.py`）：输出 patch 特征 `[B, 576, 1024]`（以 CLIP-L/336 为例，24x24=576 patch，不含 CLS）。
3. Projector `build_vision_projector`（`multimodal_projector/builder.py`）：linear 或 `mlpNx_gelu` 将 1024 → LLM hidden（如 4096 for 7B），输出 `[B, 576, hidden]`。
4. Special token 占位：文本中 `<image>` 经 `tokenizer_image_token` 变为 `IMAGE_TOKEN_INDEX=-200` 占位。
5. 融合：`prepare_inputs_labels_for_multimodal`（`llava/model/llava_arch.py`）把 `<image>` 位置替换为 projector 后的视觉 embeddings，文本 embeddings 其余位置；构造 `attention_mask`、`position_ids`、labels（图像位置 label=-100）。
6. LLM forward：`LlavaLlamaForCausalLM.forward` 调用 HF `LlamaForCausalLM`，`inputs_embeds`+`attention_mask` 进自回归；`generate` 走同样路径且支持 streaming。

---

## 2. 最小推理闭环（先跑起来再细看）

### 2.1 如何运行
- 命令示例（单卡、已下载 7B 权重到 `llava-v1.5-7b/`）：
  ```bash
  python -m llava.serve.cli \
    --model-path ./llava-v1.5-7b \
    --image-file ./images/demo.jpg \
    --device cuda --temperature 0.2 --max-new-tokens 128
  ```
- 关键参数：`--model-path` 权重目录或 HF 名称；`--model-base`（可选，LoRA/base 组合）；`--device`；`--temperature`、`--max-new-tokens`。

### 2.2 调用链（按阅读顺序）
1. CLI 入口：`llava/serve/cli.py:main`
   - 解析参数→`disable_torch_init`（跳过 Linear/LN 初始化）。
   - `load_pretrained_model`（`llava/model/builder.py`）返回 `tokenizer, model, image_processor, context_len`。
2. Model 加载：`builder.load_pretrained_model`
   - 根据路径/名字判断是否 LLaVA → 构造 `LlavaLlamaForCausalLM`，加载权重/quant，`model.get_vision_tower().load_model()`，添加特殊 token（`DEFAULT_IMAGE_PATCH_TOKEN` 等），`resize_token_embeddings`。
   - `image_processor` 来自 CLIPVisionModel（HuggingFace）。
3. 输入准备：
   - 读图：`load_image` → `process_images`（`llava/mm_utils.py`），默认 `image_processor.preprocess`，输出 `pixel_values` `[1,3,336,336]` float16。
   - Prompt 模板：`conversation.py` 根据模型名字选择 `conv_templates`（如 `llava_v1`）；首次消息前置 `<im_start><image><im_end>\n`（若配置）。
   - Tokenize：`tokenizer_image_token` 把 `<image>` 变为 `IMAGE_TOKEN_INDEX`，返回 `input_ids` `[1, seq_len]`。
4. 生成：
   - `model.generate`（`llava/model/language_model/llava_llama.py`）→ `prepare_inputs_labels_for_multimodal`（`llava/model/llava_arch.py`）：
     - `encode_images`: vision tower → projector，得到 `[B, num_patches, hidden]`。
     - 把 `<image>` 位置替换为视觉 embeddings；其余 token 用 `embed_tokens`；拼后 `new_input_embeds` `[B, seq_len + num_image_tokens, hidden]`。
     - 构造 `attention_mask`、`position_ids`（按 padding side）并裁剪到 `tokenizer_model_max_length`。
   - HF `generate` 内部 loop，返回 `output_ids`。
5. 解码：`tokenizer.decode(output_ids[0]).strip()`。

### 2.3 当场验证点（可 `print`/加断点）
- `process_images` 后：`pixel_values.shape` 预期 `[1,3,336,336]`（anyres 则 `[1, num_tiles, 3, H, W]` list）。
- `tokenizer_image_token`：查看 `input_ids[:20]` 是否包含 `IMAGE_TOKEN_INDEX=-200`。
- `prepare_inputs_labels_for_multimodal`：
  - `image_features.shape` 典型 `[1,576,4096]`（7B）；若 anyres，长度更大。
  - `new_input_embeds.shape`、`attention_mask.sum()`；确认序列长度 = 文本 token 数 + 576。
- `generate` 前：`max_new_tokens` 是否被截断为 `context_len - current_len`。
- 解码后：检查是否残留 `<im_start>` 等特殊 token。

---

## 3. 模型结构拆解

### 3.1 Vision Encoder（视觉编码器）
- 位置：`llava/model/multimodal_encoder/clip_encoder.py`
- 实现：`CLIPVisionTower` 包装 HF `CLIPVisionModel`；`feature_select` 取 `hidden_states[select_layer]`，`select_feature='patch'` 默认丢 CLS。
- 预处理：`mm_utils.process_images` 调用 `CLIPImageProcessor.preprocess`（resize/crop/normalize），支持 `pad`（方形扩展）、`anyres`（分块）。
- 输出：默认 CLIP-L/336 → `[B,576,1024]`（patch tokens）；`hidden_size` 由 config。
- 调试：在 `CLIPVisionTower.forward` 打印 `image_forward_outs.hidden_states[-1].shape`。

### 3.2 Projector / Adapter
- 位置：`llava/model/multimodal_projector/builder.py`
- 类型：
  - `linear`（默认）：`nn.Linear(mm_hidden_size, hidden_size)`。
  - `mlpNx_gelu`：多层 Linear+GELU，首层 mm_hidden → hidden，后续 hidden → hidden。
  - `identity`：直通。
- 视觉 token 数量：由 vision tower patch 数决定（如 576）。`mm_patch_merge_type` 支持 `flat`（直接 flatten），`spatial-unpad` 针对 anyres。
- 配置切换：训练时 `--mm_projector_type`（`ModelArguments`），加载时从 config 读取。
- 调试：打印 projector 权重形状；forward 后 `image_features.shape`。

### 3.3 LLM（语言模型）
- 位置：`llava/model/language_model/llava_llama.py`（LLaMA 系列），`llava_mistral.py`、`llava_mpt.py` 类似。
- 加载：`load_pretrained_model` 内部调用 `LlavaLlamaForCausalLM.from_pretrained`；添加特殊 token 后 `resize_token_embeddings`。
- Tokenizer：`AutoTokenizer.from_pretrained`；若 `mm_use_im_patch_token`/`mm_use_im_start_end`，在 builder 中 `tokenizer.add_tokens`。
- 位置与 mask：`prepare_inputs_labels_for_multimodal` 构造 `attention_mask`、`position_ids`（左/右 padding 兼容）。
- Generate 参数：在 CLI 由 `--temperature`/`--max-new-tokens`；`model_worker.py` 还用 `top_p`、`num_beams`（`run_llava.py`）。
- 调试：在 `generate` 前打印 `model.config.max_position_embeddings`、`input_ids.shape`、`attention_mask.sum()`。

### 3.4 多模态融合机制
- `<image>` 解析：`mm_utils.tokenizer_image_token` 将 `<image>` 插入 `IMAGE_TOKEN_INDEX` 占位（可含 BOS）。
- 融合顺序：`prepare_inputs_labels_for_multimodal` 按文本分段→在 `<image>` 处插入视觉 embeddings（保持顺序），文本两侧保持原始顺序。
- Attention mask：文本 padding 先被去掉再重建；视觉 token 位置 mask=True；position_ids 连续递增（左填充则从后对齐）。
- Label mask：
  - 训练：图像对应位置全设 `IGNORE_INDEX=-100`；用户 prompt 位置也会按模板屏蔽（见 `preprocess_*`）。
  - 推理：labels=None，返回 None。
- “只预测助手”处理：`preprocess_v1/preprocess_llama_2` 中根据轮次将 user 部分 `target` 置 -100，assistant 回复位置保留。

---

## 4. 训练链路

### 4.1 入口与配置
- 入口：`llava/train/train.py:train`（`python -m llava.train.train ...`），`train_mem.py`（指定 `flash_attention_2`），`train_xformers.py`（xformers monkey patch）。
- 配置：
  - `ModelArguments`（vision_tower、projector_type、mm_use_im_start_end 等）。
  - `DataArguments`（data_path、image_folder、lazy_preprocess、image_aspect_ratio）。
  - `TrainingArguments` 继承 HF，新增 `mm_projector_lr`、LoRA 参数等；分布式/Deepspeed 通过 HF 自带 args + `scripts/zero*.json`。

### 4.2 数据集与样本格式
- 数据集类：`LazySupervisedDataset`（`train.py`）
  - 读取 JSON（`list_data_dict`），字段：`conversations`（多轮，含 `from`/`value`），可选 `image` 路径。
  - 读图：`PIL.Image.open`，若 `image_aspect_ratio=='pad'` 做方形填充；调用 `image_processor.preprocess` → `pixel_values` `[3,H,W]`。
  - 文本多模态处理：`preprocess_multimodal` 把数据里的 `<image>` 替换成 `DEFAULT_IMAGE_TOKEN` 或 `<im_start><image><im_end>`。
- 读取顺序：建议先看 `__getitem__` → `preprocess_multimodal` → `preprocess`。

### 4.3 Tokenization、对齐与 Label Mask
- `preprocess_*` 系列（`train.py`）
  - `preprocess_v1`（默认 conv 模板为 `SeparatorStyle.TWO`）：对每轮，把 user 部分 label 设 -100，只训练 assistant 回复。
  - `preprocess_llama_2`、`preprocess_mpt` 针对对应模板。
  - `tokenizer_image_token` 用于含图 prompt（保持 `<image>` 占位）。
  - 返回 `input_ids`、`labels`（同 shape，mask 后的 -100）。
- Collator：`DataCollatorForSupervisedDataset.__call__`
  - `pad_sequence` 到 batch 内最长，`input_ids` pad to `pad_token_id`，`labels` pad to -100；截断至 `tokenizer.model_max_length`。
  - 组装 `attention_mask=input_ids.ne(pad_token_id)`；图像若存在则 `torch.stack` 成 `[B,3,H,W]`。

### 4.4 前向、loss 与训练策略
- Forward：`LlavaLlamaForCausalLM.forward` 接收 `images`，内部 `prepare_inputs_labels_for_multimodal` 生成 `inputs_embeds` 和 `labels`，最终 `LlamaForCausalLM` 计算 `CrossEntropy`（忽略 -100）。
- Gradient checkpointing：`training_args.gradient_checkpointing` → `enable_input_require_grads` 或 hook。
- 参数冻结/学习率：
  - `freeze_backbone` 可冻结 LLM。
  - `tune_mm_mlp_adapter` 只训练 projector（其余 `requires_grad=False`）。
  - `mm_projector_lr` 在 `LLaVATrainer.create_optimizer` 单独分组 lr。
- LoRA/QLoRA：`training_args.lora_enable` 构造 `LoraConfig`（目标模块为所有 Linear，排除 mm_projector）；bitsandbytes 4/8bit 训练通过 `bits` 参数。
- Scheduler/Warmup：沿用 HF Trainer 默认（`get_scheduler`）；在 `Trainer` 内根据 `TrainingArguments` 自动设置。

### 4.5 保存/加载与权重兼容
- 保存：`safe_save_model_for_hf_trainer`（常规）或 `LLaVATrainer._save_checkpoint`（仅保存 projector/vision_resampler when tune_mm_mlp_adapter）。
  - `mm_projector.bin` 单独存；LoRA 另存 `non_lora_trainables.bin`。
- 组合加载：`load_pretrained_model` 若提供 `model_base` + `mm_projector.bin`，会加载 base LLM 再覆写 projector 权重。
- 兼容升级：旧权重可用 `llava/model/utils.py:auto_upgrade` 升级 config（v0 → 新代码）。

---

## 5. 推理与 Serving 链路

### 5.1 对话与 prompt 模板
- 模板定义：`conversation.py` 的 `conv_templates`（`llava_v1`、`llava_llama_2`、`mpt` 等）。
- 推理入口选择：`llava/serve/cli.py` 自动按模型名推断 `conv_mode`（可 `--conv-mode` 覆盖）。
- `<image>` 插入：首次轮次在 CLI/worker 中自动加 `<im_start><image><im_end>\n` 或 `<image>\n`。

### 5.2 Web/多实例 Serving
- Worker：`llava/serve/model_worker.py`
  - 载入模型同 builder；`generate_stream` 支持多图（数量需与 `<image>` 出现次数一致），使用 `TextIteratorStreamer` 逐 token 推送。
  - 停止词：`stop_str = conv.sep/sep2`；`max_new_tokens` 会减去已占用长度和图像 token。
  - 并发：`model_semaphore` 控制 `limit_model_concurrency`。
- Controller/Gradio：`llava/serve/controller.py`、`gradio_web_server.py` 组合 worker；阅读顺序：Controller 注册 → Worker 心跳 → Web 发送请求 → Worker `generate_stream`。
- sglang Worker：`llava/serve/sglang_worker.py`（替代生成后端）。

### 5.3 停止与缓存
- Stopping criteria：可参考 `mm_utils.KeywordsStoppingCriteria`（worker 默认未启用，可自定义）。
- Streaming：`TextStreamer`（CLI）/`TextIteratorStreamer`（worker）在生成线程中运行。
- 历史缓存：`prepare_inputs_for_generation` 透传 `past_key_values`（HF 负责），多轮对话由上层维护 prompt（`conv.append_message`）。

---

## 6. 评测链路

- 主脚本：`llava/eval/run_llava.py`（单轮问答/VQA 风格）。调用链与 CLI 相同，额外处理 `IMAGE_PLACEHOLDER`，支持多图（`--sep` 分隔）。
- 任务脚本示例：
  - `eval_textvqa.py`、`eval_pope.py`、`eval_science_qa.py` 等使用 `model_vqa*.py` 里的数据加载与推理包装。
  - `generate_webpage_data_from_table.py`、`table/`、`webpage/` 处理特定数据格式。
- 阅读顺序：先看 `run_llava.py`（通用模式），再按任务打开对应 `eval_*.py`，关注如何组织 prompt 和评分。

---

## 7. 调试断点与观察项

### 7.1 推理链路断点
- `llava/serve/cli.py:main`：查看解析后的 `args`、`conv_mode`。
- `mm_utils.process_images`：打印 `pixel_values.shape`、`dtype`、是否 list（anyres）。常见 bug：shape 不一致 → 预处理模式与模型 config 的 `image_aspect_ratio` 不匹配。
- `mm_utils.tokenizer_image_token`：检查 `<image>` 是否被替换为 -200；若缺失，prompt 模板可能没包含 `<image>`。
- `llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal`
  - 观察 `image_features.shape`、`cur_input_ids` 中 `<image>` 数量。
  - 查看 `new_input_embeds.shape` 和 `attention_mask.sum()`；若长度超界，`tokenizer_model_max_length` 可能太小。
- `model.generate` 调用前：`max_new_tokens` 是否被裁剪为 >0。
- 解码后：若输出残留 `<im_start>`，检查 `tokenizer.decode(..., skip_special_tokens=True)`。

### 7.2 训练链路断点
- `LazySupervisedDataset.__getitem__`：检查样本字段；`image.shape` `[3,336,336]`；`sources` 是否带 `<image>`。
- `preprocess_*`：打印某条样本的 `input_ids` / `labels`，确认 user 部分 label = -100，assistant 部分为词表 id。
- `DataCollatorForSupervisedDataset.__call__`：`batch['input_ids'].shape` `[B, T]`，`batch['labels']` 同形状，pad 位置应为 -100。
- `prepare_inputs_labels_for_multimodal`（训练 forward）：确认视觉 token label=-100；`attention_mask` 长度=embed 长度。
- Loss 调试：`outputs.loss` 是否为 NaN；若 NaN，检查 `image_features` dtype（应 float16/bfloat16），以及是否有空 prompt 导致全 -100。
- Optimizer step：在 `LLaVATrainer.create_optimizer` 处打印 lr 分组，确认 projector 是否使用独立 lr。

### 7.3 常见现象 → 可能原因 → 去哪里查
- 生成全空/重复：`attention_mask` 或 `<image>` 未插入 → 查 `prepare_inputs_labels_for_multimodal` 输入；prompt 未含 `<image>`。
- 输出乱码/特殊符号：Tokenizer 未加载正确或未 `resize_token_embeddings` → 查 `load_pretrained_model` log。
- 序列过长报错：`max_new_tokens` 超过上下文 → 在 worker 中 `max_new_tokens` 已被截断，确认 `prompt` 过长；必要时减少图像分辨率或 anyres patch 数。
- Loss 恒定不变：可能 projector 冻结或仅图像 label=-100 → 检查 `tune_mm_mlp_adapter`、`requires_grad`，以及 label mask。

---

## 8. 自测清单
- 我能解释 `<image>` token 如何在 `mm_utils.tokenizer_image_token` → `prepare_inputs_labels_for_multimodal` 中映射到视觉 embeddings（路径、函数名、shape）。
- 我能手算一次序列长度：文本 token 数 + 576（或 anyres patch 数）并在 `new_input_embeds.shape` 中验证。
- 我能说明 label mask 的生成规则（user/图像部分为 -100），并在 `DataCollatorForSupervisedDataset` 输出的一个 batch 上打印确认。
- 我能修改 projector（例如将 `mm_projector_type` 设为 `mlp3x_gelu`）并重新训练/推理，知道需要在哪些文件改（config/args）并观察 `mm_projector` shape。
- 我能定位一次输出异常（例如残留 `<im_start>`）：先检查 tokenizer decode，再检查 prompt 模板，再到 `prepare_inputs_labels_for_multimodal`，最后检查生成的 `stop_str`。

---

### 推荐阅读顺序总览
1. 跑通最小 CLI：`llava/serve/cli.py` → `builder.load_pretrained_model` → `prepare_inputs_labels_for_multimodal`。
2. 拆 Vision & Projector：`multimodal_encoder/clip_encoder.py`、`multimodal_projector/builder.py`。
3. 看融合与 forward：`llava/model/llava_arch.py`、`language_model/llava_llama.py`。
4. 训练链路：`train.py` 数据集/预处理/Trainer → `train_xformers.py`/`train_mem.py`。
5. Serving 与评测：`serve/model_worker.py`、`eval/run_llava.py`，根据需要深入其它 `eval_*`.

