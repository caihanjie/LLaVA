# LLaVA CLI 运行与图文融合数据流详解

命令示例：
```
python llava/serve/cli.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-file /home/chj/LLaVA/images/llava_logo.png \
  --load-4bit
```

## 启动/加载流程
- 入口：`main` 解析参数，调用 `disable_torch_init()`（覆盖 Linear/LayerNorm.reset_parameters，避免重复初始化，减速显存占用）。
- 模型识别：`get_model_name_from_path`→`llava-v1.5-7b`，`conv_mode` 自动选 `llava_v1`（两分隔符，角色 `USER/ASSISTANT`，system 为通用助手提示）。
- 加载模型 `load_pretrained_model`：
  - 4bit 量化：`BitsAndBytesConfig(load_in_4bit=True, nf4, bnb_4bit_use_double_quant=True, compute_dtype=float16)`。
  - Tokenizer：从 `model_path` 直接加载；未加入图像特殊 token（配置 `mm_use_im_patch_token=False, mm_use_im_start_end=False`）。
  - 模型：`LlavaLlamaForCausalLM`（LLaMA 7B，32 层、32 头、KV 头 32，hidden_size=4096，max_position_embeddings=4096，pad_token_id=0，bos=1，eos=2）。
  - 视觉塔：`openai/clip-vit-large-patch14-336`，取倒数第二层（`mm_vision_select_layer=-2`），特征类型 `patch`（去掉 CLS）。
  - 投影头：`mm_projector_type="mlp2x_gelu"` → Linear→GELU→Linear，将 CLIP hidden 映射到 4096。
  - 配置视觉：`image_aspect_ratio="pad"`；`mm_patch_merge_type=None` 但代码默认 `'flat'`，即不做 spatial 重排。

## 图片→像素张量
- `load_image` 读取本地/URL 图片为 RGB，记录 `image_size=(W,H)`。
- `process_images`：因 `image_aspect_ratio='pad'`，先将图像扩展成方形（背景为 image_mean*255），再用 `CLIPImageProcessor` 预处理（resize/crop/normalize）。
- 输出形状 `(1,3,336,336)` 的 `pixel_values`，再 `to(model.device, float16)`；如传多图会组成 list 或 batch。

## 首轮 prompt 构造
- 首次用户输入前若 `image` 非 None：根据 `mm_use_im_start_end=False`，在用户文本前插入字符串 `<image>\n`，并将 `image=None` 以防后续重复。
- 对话模板 `llava_v1`（双分隔符）：`system`+`USER: <image>\n<question> </s>ASSISTANT:`。

## `<image>` 占位符→特殊 token
- 调用 `tokenizer_image_token(prompt, IMAGE_TOKEN_INDEX=-200)`：
  - `prompt.split('<image>')`，各段分别 tokenizer。
  - 若首段首 token 是 BOS，保留 BOS（offset=1）。
  - 在段间插入 `IMAGE_TOKEN_INDEX`，得到形如 `[1, ..., -200, ...]` 的 `input_ids`，再 batch 维 `unsqueeze(0)`。

## 视觉特征生成
- `generate` 内部调用 `prepare_inputs_labels_for_multimodal`：
  - `encode_images`：视觉塔前向，得到 `image_features_raw` 形状 `(B, 576, hidden_clip)`（336/14=24，24*24=576 patch）。
  - 过 `mm_projector(mlp2x_gelu)` → `(B, 576, 4096)`，dtype 与模型一致。

## 图文融合核心（占位 token 替换）
文件：`llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal`

1. 预处理掩码/位置：若无 `attention_mask`，新建全 True；`position_ids` 默认为 `arange(seq_len)`；`labels` 缺省时填充 `IGNORE_INDEX=-100`。
2. 删除 padding：`input_ids = input_ids[mask==1]` 按样本裁剪，`labels` 同步。
3. 遍历 batch，针对每条样本：
   - 统计图像数：`num_images = (input_ids == -200).sum()`。
   - 若 `num_images==0`：直接词嵌入全部文本，附加空的图像特征（占位为空）。
   - 若有图像：
     - 构造分界下标 `image_token_indices = [-1, idx1, idx2, ..., seq_len]`（即你提到的结构），用于切分文本区段。
     - 文本拆分：`cur_input_ids_noim[i] = tokens between image_token_indices[i]+1 and image_token_indices[i+1]`，标签同样拆分。
     - 词嵌入：把拆分后的文本拼接嵌入，再按原分段 `split` 回来。
     - 重新交织：对每个区段 `i` 追加文本嵌入+标签；若 `i < num_images`，再插入对应 `image_features[cur_image_idx]`（shape `(576,4096)`）以及长度匹配的标签全 `IGNORE_INDEX`。`cur_image_idx` 逐图递增。
     - 最终序列：文本块与图像 patch 序列交替，原本单个 `IMAGE_TOKEN_INDEX` 被整段 576 patch 向量取代。
4. 序列裁剪：若配置 `tokenizer_model_max_length`（本模型无），则截断过长；否则保留完整交织序列。
5. Padding 对齐：取 batch 内最大长度 `max_len`，右填充（默认 padding_side='right'）0 向量，构造：
   - `new_input_embeds_padded`: `(B, max_len, 4096)`，有效部分为文本/图像嵌入，填充段为 0。
   - `new_labels_padded`: 其余位置为 `IGNORE_INDEX`。
   - `attention_mask`: True 表示有效 token；`position_ids`: 从 0 递增（填充处为 0）。
6. 返回给 `generate`：`inputs_embeds=new_input_embeds_padded`，`position_ids/attention_mask` 与之对齐，`input_ids=None`（后续直接用嵌入生成）。

## 生成与流式输出
- `model.generate`（继承 transformers）：
  - 采样：`temperature=0.2` → `do_sample=False`，贪婪/beam 默认；`max_new_tokens=512`；`use_cache=True`。
  - 传入 `images=image_tensor, image_sizes=[(W,H)]` 仅用于准备嵌入，之后生成阶段不再使用原始图像。
  - `TextStreamer` 实时打印生成 token（跳过 prompt、特殊 token）。
  - `stop_str`/`keywords` 仅用于 streamer 终止判断；未显式传入 `stopping_criteria`，依赖默认 EOS。
- 输出解码：`tokenizer.decode(output_ids[0]).strip()`；写回 `conv.messages`，若 `--debug` 打印 prompt+outputs。

## 重要配置摘要（llava-v1.5-7b/config.json）
- 语言：`hidden_size=4096, num_layers=32, num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=4096, rms_norm_eps=1e-5, pad_token_id=0, bos=1, eos=2, rope_theta=None`（使用 RoPE 默认）。
- 视觉：`mm_vision_tower=openai/clip-vit-large-patch14-336`, `mm_vision_select_layer=-2`, `mm_vision_select_feature=patch`。
- 融合：`mm_projector_type=mlp2x_gelu`, `mm_use_im_patch_token=False`, `mm_use_im_start_end=False`, `image_aspect_ratio=pad`, `mm_patch_merge_type` 缺省→默认 `'flat'`，每图 576 patch 嵌入插入。

## 注意事项与扩展
- 多图支持：`process_images` 返回 list/batch 时，`prepare_inputs_labels_for_multimodal` 会按 `IMAGE_TOKEN_INDEX` 顺序为每张图插入对应特征。
- 如果换用 `image_aspect_ratio='anyres'` 或 `mm_patch_merge_type` 含 `spatial/unpad`，会走空间重排/去 padding 路径，patch 排布与插入长度发生变化。
- 若改用 `mm_use_im_start_end=True`，需要在 prompt 手动包含 `<im_start><image><im_end>` 序列并在 tokenizer 里注册这些 token。
- 量化加载只影响权重存储/乘法精度，不改变上述图文融合逻辑。
