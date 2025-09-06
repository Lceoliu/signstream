# 项目说明文档

**项目代号**：SignStream-RVQ

**面向代理**：请用原生 PyTorch + HuggingFace 生态（`transformers`/`accelerate` 可安装但第一步不必使用）实现本说明文档所述代码。**禁止**使用 Lightning/DeepSpeed/庞大训练框架。兼容 torch.distributed 的 DDP 即可。

**数据**：已提供 **CSL-Daily** 数据集及其数据格式 CSL-Daily_README。

**目标阶段**：**第一步（Stage 1）——多路 RVQ 离散器与数据/训练脚手架**。

---

## 0. 项目总目的（长线）

构建一个**实时**手语翻译系统：将 3D pose 时间序列离散为**sign tokens**，作为 **decoder-only LLM**（如 Qwen）的**前缀条件**，以 **KV cache** 支持的方式**流式**生成文本。系统核心包含三部分：

1. **多路 RVQ 离散器（本阶段实现）**：将面部/左右手/身体子流的 pose 表示离散为 token，输出语义稳定、时序平滑、码本利用率健康的离散序列。
2. **Prefix-LM 适配（后续阶段）**：将 sign tokens 作为 LLM 的前缀，<SEP> 后仅对文本计算语言建模损失。
3. **流式推理与<COMMIT>稳定机制（后续阶段）**：滑窗 KV + 摘要 token + 提交点，保证“边看边说、少回改”。

---

## 1. 第一阶段的目标（本次要交付的“能跑起来”的最小系统）

### 1.1 业务目标

- 在 CSL-Daily 上**训练一个可用的多路 RVQ 离散器**（Residual VQ，2–3 级），输入为对齐到统一坐标系/帧率的 2D pose 序列（面部/左手/右手/躯干/全身五路）。
- 产出**符号序列**，并支持将每个时间块（chunk）格式化为**可供 LLM 使用的 token 模板**（仅定义与导出，不做 LLM 训练）。
- 提供**指标与可视化**：重构损失、时序一致性、码本使用熵/困惑度、每块 token 数、运行时延统计。
- 提供**推理脚本**：给定一段样本，导出其离散 token 序列与简易 Run-Length 压缩（No-Change 标记）。
- 代码结构**清晰、可扩展**，为后续 Stage 3/4（Prefix-LM & 流式）直接接入做准备。

### 1.2 技术约束

- **框架**：PyTorch 2.1.0。
- **本机环境**: 已经配置了一个包含大多数所需库的 python conda 环境，请使用 conda activate SLR_realtime 来激活。
- **禁止** Lightning/DeepSpeed 以及将训练循环交给“黑盒 Trainer”。
- **配置**：YAML 配置文件 + 命令行覆盖；保证可复现实验。
- **日志**：TensorBoard.
- **设备**：优先单卡 3090 24G，保证后续多卡训练的可行性。

---

## 2. 数据假设与输入输出

> 以“CSL-Daily 数据格式 README”为准。CSL-Daily root dir 绝对位置位于/workspace/data/public_data/CSL_Daily, README 位于./CSL-Daily_README.md

> 在本目录下有./CSL-Daily 软链接，所以可以直接访问

### 2.1 输入（训练/推理）

- **样本单位**：一段视频对应的 **2D pose 序列**，参考 COCO 关键点格式。
- 按照数据集 README 所给的格式将输入按身体部位分区为不同的子流输入：face, left_hand, right_hand, body, full_body(包含以上所有部位)
- 必须阅读 CSL-Daily_README.md!

### 2.2 预处理（已实现部分，请参考并添加或修正）

- **时域**：统一帧率；线性插值/低通滤波；可选时序截断到最大长度，注意 collate 时候的 padding！
- **归一化**：各分区以 CSL-Daily_README 给定的中心定义局部坐标系，所有子流坐标各自归一化；必须按骨长/下眼睑长归一单位长度；在这之后，将各个分区的 2d 点坐标缩放至[-8, 8]的 bounding box 中。
- **块化（chunking）**：将 T 帧划分为长度 `L` 的块（建议 8–12 帧/块，或 320–400ms/块）；每块将在各子流上独立编码。
- **数据增强**：镜像（左右手交换需同步）、速度增强（随机加速或减速）、关节遮挡 dropout（可开关）。

### 2.3 输出（本阶段）

- **离散 token 序列**（仅 sign 部分）：为每个时间块产出若干 code id。
- **格式化模板**（供后续 LLM 使用）：

  ```
  <F:c12><LH:a37><RH:b08><B:d91><FB:e45>
  ```

  其中 `F/LH/RH/B/FB:` 前缀区分子流来源；`a/b/c/d/e` 为各子流所属码本的 id 空间（避免与文本词表冲突）。

- **可选压缩**：相邻块若“变化极小”，输出 `<LH:NC>xK` 等 No-Change/Run-Length 记号（简化到 JSON 序列即可）。

---

## 3. 模块与目录结构

```
signstream/
  configs/
    default.yaml                  # 全局配置（数据路径/模型/训练/日志）
  data/
    datasets.py                   # CSL-Daily Dataset & DataModule  (已部分实现)
  models/
    rvq/
      encoder.py                  # 子流编码器（Face/LH/RH/Body/FullBody, shared backbone+type embed）
      quantizer.py                # 多级 RVQ（EMA/commitment/usage 正则）
      decoder.py                  # 重构解码器（必须与encoder对称！）
      rvq_model.py                # 组合：Encoder -> RVQ -> Decoder
    metrics/
      recon.py                    # MSE/Huber/SSIM（可选）
      temporal.py                 # 时序一致性（相邻块差分）
      codebook.py                 # 使用率/困惑度/熵

  training/
    train_rvq.py                  # 训练主脚本（Accelerate）
    train_rvq_multiple_gpu.py     # 多卡训练脚本（torch.distributed）
    loop_rvq.py                   # 训练/验证循环（手写）
    optim.py                      # 优化器/调度器
    losses.py                     # 重构/commitment/usage/时序一致性/InfoNCE（留接口）
    utils_tb.py                   # tensorboard 日志/可视化
    seed.py                       # 可复现
  inference/
    export_tokens.py              # 离散推理：pose->tokens(JSON)
    rle.py                        # No-Change/Run-Length
    viz.py                        # 可视化：token时间轴/码本热图
    visualizer.py                 # 可视化工具（对比可视化原2d点视频以及sign token可视化视频，即可视化重构器效果）
  io/
    dataloader_utils.py           # 多进程/缓存 （Optional）
  tests/
    test_quantizer.py             # 量化/反量化一致性
    test_dataset.py               # 数据切块/归一化正确性
    test_export.py                # 导出 token 与 RLE 的稳定性
  README.md
requirements.txt
train.sh                        # 单卡训练启动脚本
train_multi.sh                  # 多卡DDP训练启动脚本
export_tokens.sh                # 导出所有sign tokens脚本
decode_sample.sh                # 随机挑选一条视频进行encode-rvq-decode可视化对比
```

---

## 4. 模型设计（本阶段）

### 4.1 编码器/解码器

- **子流共享骨干**：使用一个共享轻量 Transformer（2-3 layer, 256 latent dim, 可配置），将 `type embedding`（Face/LH/RH/B/FB）拼接/相加/Fusion 融合，以减少参数且鼓励跨域组合表示。
- **时间聚合**：对块内帧做平均池化或轻量时序注意力（可配置），得到块级 latent 向量 `z_chunk`（每子流一份）。
- **解码器**：镜像结构，重构回块内帧的关键点（或其增量）。
- 支持可选 **`torch.compile`**。

### 4.2 RVQ 量化器

- **层数**：2–3 级残差码本（可配置）；**码本大小**：每级 1024（可配置）；latent dim d=256；EMA=0.99；每路平均每块 1~3 个 code，总计 4~8 个 code/块。
- **优化**：AdamW，lr=3e-4，wd=0.01；batch 依显存自适应；AMP 打开
- **损失**：

  - 重构：MSE/Huber（可配置）。
  - Commitment/Codebook：标准 VQ-VAE/RVQ 公式（EMA 版）
  - **usage 正则**（鼓励均匀使用，避免塌缩）: L_usage = λ_u \* (KL(u || uniform))。
  - **时序一致性**：相邻块 latent 的差分正则（或对齐到速度域）。
  - **可选 InfoNCE 钩子**：为后续语义对齐预留接口（本阶段可先留空/关闭）。
  - 总损失：L = L_rec + λ_vq L_vq + λ_tem L_tem + λ_u L_usage

- **指标**：码本困惑度、码本占用率直方图、平均每块 symbol 数、压缩率。

- **可视化代码**: 重构 decoder 的重建效果可视化。

- **健康度检查**:

  训练 50~200 epoch，在计算 validation loss 时，必须分别记录每一个 loss 的具体值，并同时进行以下验证：

  - 每 5 epoch（可配置）计算一次 codebook 利用率
  - 每 1 epoch sample 5 条(可配置) 输入数据，并输出此数据对应的 sign tokens

---

## 5. 训练与评估

### 5.1 训练配置（默认建议）

- 块长度：`L = 8–12` 帧；批大小按显存自动调整。
- 优化器：AdamW；学习率与权重衰减从配置读取。
- 训练时长：以“重构收敛 + 使用率稳定”为准（示例 100–300 epoch）。
- 混合精度：bf32/fp32（`accelerate` 自动管理）。
- 自动保存：自动保存 best checkpoint 以及每 5 epoch（可配置） 保存一次 checkpoint。
- 存储路径：在当前路径下创建 checkpoints 文件夹，且每次运行都在 checkpoints 中创建一个`${experiment_name}/${timestamp}`命名的子文件夹，用于存储 checkpoints，tensorboard event，logs 以及本次训练使用的 yaml 配置文件的拷贝等。

### 5.2 评估指标（validation）

- **重构质量**：MSE/Huber（四路分别与加权求和）。
- **时序平滑**：相邻块重构差分的均值/分位数。
- **码本健康度**：使用率、困惑度、熵；空置条目比例。
- **效率**：每秒样本数、单步延迟（ms）、显存峰值。
- **导出健壮性**：随机样本导出 token 序列成功率；RLE 后长度统计。

### 5.3 验收/单元测试

- `tests/test_quantizer.py`：前向量化+反量化误差在阈内；EMA/commitment 梯度路径稳定。
- `tests/test_dataset.py`：块化后总帧覆盖率 100%，镜像与左右手交换一致。
- `tests/test_export.py`：RLE 与还原一致性；极端“全静止/全变化”样本正确处理。
- **过拟合小样本**：取 16 个样本训练，重构误差显著下降（作为 sanity check）。

---

## 6. 配置与命令行

### 6.1 `configs/default.yaml`（示意）

```yaml
experiment_name: "Baseline-Test001"
save_path: "/path/to/save_dir"

data:
  root: "/path/to/CSL-Daily"
  split_file: "/path/to/split_file.json"
  fps: 30 # 以CSL-Daily_README中具体数据为准
  chunk_len: 10
  augment:
    mirror: true
    time_warp: true
    dropout_prob: 0.05

model:
  encoder_layer: 2
  encoder_latent_dim: 256
  latent_dim: 256
  type_embed_dim: 16
  rvq:
    levels: 3
    codebook_size: 2048
    ema_decay: 0.99
    commitment_beta: 0.25
    usage_reg: 1e-3
    temporal_loss_alpha: 0.05

train:
  epochs: 200
  batch_size: 16
  lr: 3e-4
  wd: 0.01
  amp: "bf32"
  log_wandb: false
  seed: 42

export:
  enable_rle: true
  rle_threshold: 0.02 # 块间变化阈值
```

### 6.2 典型命令

```bash
# 训练
accelerate launch signstream/training/train_rvq.py --config configs/default.yaml

# 多卡训练
CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun --nproc_per_node=4 --nnodes=1 signstream/training/train_rvq_multiple_gpu.py --config configs/default.yaml

# 断点续训
accelerate launch signstream/training/train_rvq.py --config configs/default.yaml --checkpoint path/to/ckpt.pt

# 验证/导出
python signstream/inference/export_tokens.py \
  --config configs/default.yaml \
  --checkpoint path/to/ckpt.pt \
  --split val \
  --num-samples 10 \
  --out ./exports/tokens.jsonl
```

---

## 7. 导出格式（供后续 LLM 阶段使用）

### 7.1 JSON Lines（每行一条样本）

```json
{
	"video_name": "xxxx",
	"fps": 25,
	"chunk_len": 10,
	"tokens": [
		{ "t": 0, "F": [12, 7], "LH": [37], "RH": [8, 8], "B": [91] },
		{ "t": 1, "F": [7], "LH": ["NC", 3], "RH": [8], "B": [13] }
	],
	"rle": true,
	"meta": { "note": "ids按各自码本空间编码，不与文本词表混用" }
}
```

> 说明：`"LH":["NC",3]` 表示左手 3 个块无显著变化的 RLE 压缩。

### 7.2 Token 模板（示例，仅供可视化/调试）

```
<T0><F:c12><F:c7><LH:a37><RH:b8><RH:b8><B:d91><FB:e45>
<T1><F:c7><LH:NCx3><RH:b8><B:d13><FB:e87>
```

---

## 8. 非功能性要求

- **可复现**：设定随机种子；保存 `config.yaml`、`git hash`、环境依赖。
- **容错**：数据异常（缺帧/NaN）需跳过并记录。
- **性能**：DataLoader 多进程、pin memory；大样本可分块载入。
- **代码质量**：PEP8 基本规范，关键模块含文档字符串与类型注解。

---

## 9. 里程碑与交付物

- **M0（脚手架完成）**：目录/配置/数据管道/训练循环（空模型可跑通）。
- **M1（RVQ 可训练）**：重构损失下降，码本使用率>30%，困惑度>100（示例阈）。
- **M2（导出与可视化）**：能稳定导出 token 序列 + RLE 可选；提供时序条带图与码本热图。
- **M3（报告）**：在验证集产出指标：MSE 曲线、使用率直方图、平均每块符号数、导出长度统计。

**交付物**：源码仓（含 `README.md`、`configs/default.yaml`）、1–2 个训练检查点、`exports/tokens.jsonl` 示例、日志与可视化截图。

---

## 10. 未来扩展挂钩（非本阶段实现）

- **Prefix-LM 接口**：保留 `signstream/inference/export_tokens.py` 的 `--to-llm` 选项，未来可直接拼接 `<SEP>` 后文本并构造 `labels=-100` 掩码。

---

## 11. 依赖与环境

使用 SLR_realtime conda 环境即可，若缺失库，直接安装并记录即可。

---

## 12. 成功标准（本阶段）

- 训练在 CSL-Daily 上可运行到收敛，**重构误差显著下降**。
- 码本**无塌缩**（使用率分布不极端），困惑度/熵稳定。
- 能对任意验证样本**稳定导出 token**；RLE 在静态片段显著压缩（>30%）。
- 所有单元测试通过；日志/可视化齐全；README 能指导他人复现。

---

如上，即为第一步的完整开发说明。请严格对照实现，并优先保证训练循环与导出流程的**简洁、稳定、可扩展**。后续阶段将直接在该脚手架上接入 LLM 前缀训练与流式解码。
