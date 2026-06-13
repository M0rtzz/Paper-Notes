---
title: >-
  [论文解读] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers
description: >-
  [CVPR2026][图像生成][Transformer] 提出 ELIT（Elastic Latent Interface Transformer），在 DiT 中插入可变长度的潜变量接口（latent interface）和轻量 Read/Write 跨注意力层，使单一模型能在推理时动态调节计算预算，同时将计算非均匀地分配到图像中更难的区域，在 ImageNet 512px 上 FID 最高降低 53%。
tags:
  - "CVPR2026"
  - "图像生成"
  - "Transformer"
  - "弹性推理"
  - "潜变量接口"
  - "自适应计算"
  - "多预算模型"
  - "跨注意力"
---

# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**会议**: CVPR2026  
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)  
**代码**: [snap-research/elit](https://snap-research.github.io/elit)  
**领域**: 图像生成  
**关键词**: 扩散 Transformer, 弹性推理, 潜变量接口, 自适应计算, 多预算模型, 跨注意力

## 一句话总结

提出 ELIT（Elastic Latent Interface Transformer），在 DiT 中插入可变长度的潜变量接口（latent interface）和轻量 Read/Write 跨注意力层，使单一模型能在推理时动态调节计算预算，同时将计算非均匀地分配到图像中更难的区域，在 ImageNet 512px 上 FID 最高降低 53%。

## 背景与动机

1. **DiT 计算刚性**：Diffusion Transformer 的每步 FLOPs 固定为输入分辨率的函数，无法根据延迟/质量需求灵活调节
2. **均匀计算分配浪费资源**：DiT 对所有空间 token 均匀分配计算，但图像中不同区域的生成难度差异很大（简单背景 vs 复杂纹理）
3. **零填充实验佐证**：作者用零值 patch 填充图像增加 token 数，DiT-B/2-Synth 的 FID 并未提升，说明 DiT 无法将多余计算转移到有信息的区域
4. **现有方法各有局限**：Masking 方法（MaskDiT、MDTv2）加速训练但推理时仍需全 token；训练无关加速方法不改善训练效率；自适应生成器（FlexiDiT）仍均匀分配计算或复杂度过高
5. **可变长度表征止步于 autoencoder**：FlexTok、TiTok 等在编码器阶段学习可变长度表征，但未将弹性延伸到生成模型内部
6. **RIN/FIT 偏离 DiT 架构**：潜变量 token 方法（RIN、FIT）虽能非均匀分配计算，但大幅改变架构设计，阻碍了主流 DiT 生态的采用

## 方法详解

### 整体框架

ELIT 要解决 DiT 的两个刚性问题：每步 FLOPs 被分辨率写死、无法按需调节，而且对所有空间 token 均匀分配计算、白白浪费在简单背景上。它的办法是在标准 DiT 里塞进一段“可变长度的潜变量接口”，让推理时的潜变量 token 数 $K$ 变成一个用户旋钮，直接决定每步算力。整体是三段式结构：输入先过 Spatial Head（$B_{\text{in}}$ 个 block）提取空间特征；Read 层用轻量跨注意力把空间信息抽进 $K$ 个潜变量 token（优先关注高损失的困难区域）；主算力放在 Latent Core（$B_{\text{core}}$ 个标准 block）的潜变量域里；Write 层再把更新后的潜变量广播回空间 token；最后 Spatial Tail（$B_{\text{out}}$ 个 block）恢复细节、输出速度预测。调大或调小 $K$，就在同一个模型里平滑地换算力预算。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入潜变量 X_t + 时间步 t"] --> B["Spatial Head<br/>B_in 个标准 block 提空间特征"]
    B --> C["Read 跨注意力<br/>K 个潜变量 token 为 Query 抽空间信息<br/>分组跨注意力降复杂度"]
    C --> D["Latent Core<br/>B_core 个标准 block 在潜变量域算主算力"]
    D --> E["Write 跨注意力<br/>更新后潜变量广播回空间 token"]
    E --> F["Spatial Tail<br/>B_out 个 block 恢复细节"]
    F --> G["输出：速度预测"]
    K["尾部随机丢弃<br/>训练随机采样 K、推理调 K 换算力预算"] -.控制.-> C
    G -.低预算弱模型做引导.-> H["Cheap CFG<br/>主项满预算 + 引导项低预算"]
```

### 关键设计

**1. Read/Write 跨注意力：用两层对称跨注意力打通空间域与潜变量域**

弹性的前提是空间 token 和可变数量的潜变量 token 之间能高效交换信息。Read 以潜变量为 Query、空间 token 为 Key/Value 做跨注意力 + MLP，把信息“读”进接口；Write 完全对称地把信息“写”回空间。两层都用 pre-norm + adaLN-Zero 调制保持时间步感知，并加 QK 归一化稳住训练。正因为只新增这两层、不改 RF 训练目标和 DiT 主体，ELIT 才能即插即用地套到 DiT/U-ViT/HDiT/MM-DiT 上。

**2. 分组跨注意力：把 Read/Write 的复杂度从 $\mathcal{O}(NK)$ 压到 $\mathcal{O}(NK/G)$**

当空间 token 数 $N$ 和潜变量数 $K$ 都不小时，全局跨注意力会成为瓶颈。ELIT 把空间 token 划成 $G$ 个不重叠组（如 4×4 网格），每组只分到 $J = K/G$ 个潜变量 token，跨注意力只在组内进行，复杂度因此降到 $\mathcal{O}(NK/G)$。分组大小还和分辨率匹配——256px 上 4×4 最优、512px 上 8×8 最优，逐 token（1×1）或全图都更差，说明合适的局部性本身就是有用的先验。

**3. 尾部随机丢弃：用一次训练换来推理时任意预算的弹性**

要让单一模型支持多预算，不能为每个 $K$ 单独训。ELIT 在训练时每次迭代从 $\text{Uniform}\{J_{\min}, \ldots, J_{\max}\}$ 采样实际保留的潜变量数 $\tilde{J}$、丢弃尾部 token。这样靠前的 token 被训练得更频繁、自然形成重要性排序——前面的 token 负责全局结构、后面的精修细节，推理时砍掉尾部就是平滑降预算。消融也证实这种重要性排序有效：尾部丢弃在 25% token 时 FID 36.3，明显优于随机丢弃的 38.6。

**4. Cheap Classifier-Free Guidance：让多预算特性顺手白送一个引导项**

普通 CFG 需要跑两遍完整模型，成本高。ELIT 利用自身的多预算能力，主项用完整预算 $\tilde{J}$、引导项改用低预算 $\tilde{J}_w$ 且去掉类别条件，等于免费得到一个“弱模型”版本，把 AutoGuidance 和 CFG 的好处结合起来。结果是推理 FLOPs 减少约 33% 而质量反而更优，且无需任何额外训练或手工设计的退化模型。

### 损失函数 / 训练策略

直接用标准 Rectified Flow 损失，无任何辅助损失：

$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, \mathbf{X}_1, \mathbf{X}_0} \| \mathcal{G}(\mathbf{X}_t, t) - (\mathbf{X}_1 - \mathbf{X}_0) \|_2^2$$

其中 $t$ 按 logit-normal 分布采样。多预算训练时为补偿低预算迭代省下的算力，把 batch size 从 256 增到 384 以保持训练 FLOPs 可比。训练 500k 步（图像）/ 200k 步（视频），学习率 $10^{-4}$，10k warmup，梯度裁剪 1.0，EMA $\beta=0.9999$。

## 实验关键数据

### ImageNet-1K 主实验（Table 1）

| 模型 | 分辨率 | FID↓ (–G/+G) | FDD↓ (–G/+G) | IS↑ (–G/+G) |
|------|--------|---------------|---------------|--------------|
| DiT-XL | 512px | 18.8 / 9.5 | 339.2 / 233.6 | 53.0 / 86.4 |
| ELIT-DiT MB | 512px | **10.1** / **4.5** | **164.1** / **98.2** | **88.8** / **147.0** |
| U-ViT-XL | 512px | 11.6 / 5.3 | 202.7 / 125.9 | 72.5 / 117.2 |
| ELIT-U-ViT MB | 512px | **7.7** / **3.8** | **135.8** / **83.1** | **98.0** / **159.3** |
| HDiT-XL | 512px | 13.0 / 6.0 | 260.3 / 170.5 | 69.4 / 114.2 |
| ELIT-HDiT MB | 512px | **9.6** / **4.6** | **171.2** / **106.8** | **94.7** / **154.6** |

512px 下 ELIT-MB 相对 DiT/U-ViT/HDiT 的 FID 改善分别为 **53%、28%、23%**。

### 视频生成（Kinetics-700, Table 2）

| 模型 | FID↓ (–G/+G) | FDD↓ (–G/+G) | FVD↓ (–G/+G) |
|------|---------------|---------------|---------------|
| DiT-XL | 14.0 / 11.3 | 371.5 / 309.1 | 135.9 / 100.5 |
| ELIT-DiT | **13.3** / **10.7** | **277.4** / **222.0** | **116.5** / **90.5** |

### 消融实验亮点

- **分组大小**：4×4 组在 256px 最优，8×8 在 512px 最优，组大小为 1×1（逐 token）或全图时性能下降
- **块分配**：DiT-XL 最优配置为 4-20-4（head-core-tail），约 71% block 在潜变量核心；DiT-B 最优为 3-6-3（约 67%）
- **尾部丢弃 vs 随机丢弃**：尾部丢弃（importance ordering）FID 26.6 优于随机丢弃的 27.0，25% token 时差距更大（36.3 vs 38.6）
- **收敛加速**：ELIT-DiT 在 256px/512px 分别实现 3.3×/4.0× 的收敛加速
- **大规模验证**：在 20B 参数的 Qwen-Image (MM-DiT) 上微调 120k 步，ELIT 最高可减少 63% FLOPs（约 2.7× 加速），DPG-Bench 平均分从 90.45 仅降至 88.02
- **与 TeaCache 兼容**：ELIT 与训练无关加速方法 TeaCache 可叠加使用，获得与 DiT 相当的额外加速比
- **模型规模扩展**：从 DiT-S/4 到 DiT-XL/2 全面优于 DiT，越大的模型增益越明显，额外开销占比越小

## 亮点

- **极简设计，即插即用**：仅添加 Read/Write 两个跨注意力层，保持 RF 训练目标和 DiT 架构不变，适用于 DiT/U-ViT/HDiT/MM-DiT 四种架构
- **推理时单一模型多预算**：通过控制潜变量 token 数实现平滑的质量-计算权衡，512px 下多达 60 个预算点可选
- **自适应计算分配**：Read 注意力自动聚焦高损失区域，实现非均匀计算分配，零填充实验清晰地证明了这一点
- **免费获得 AutoGuidance**：多预算特性天然提供弱模型版本，CCFG 减少约 33% 推理成本且质量更优，无需额外训练或手工设计模型退化
- **显著的收敛加速**：ELIT-DiT 训练收敛速度提升 3.3-4.0×，在相同训练 FLOPs 下大幅领先基线
- **实验覆盖全面**：图像（256px/512px）+ 视频（Kinetics-700）+ 大规模 MM-DiT（20B Qwen-Image）+ 多架构 + 详尽消融
- **动机实验说服力强**：零填充对比实验优雅地证明了 DiT 的均匀计算问题，attention map 可视化直观展示了 ELIT 的计算重分配效果

## 局限与展望

1. **大规模从头训练未验证**：Qwen-Image 实验是微调+蒸馏设置，从头训练 20B 级模型的收益尚不明确，可能面临不同的收敛挑战
2. **CCFG 饱和问题**：CCFG 倾向于比 CFG 更快地使图像饱和，在某些场景下可能导致色彩过度
3. **未在文本到图像开放域评估**：实验主要在 ImageNet 类别条件下进行，text-to-image 的效果有待验证
4. **跨步预算调度未探索**：不同采样步可能需要不同计算预算（早期步需要更少计算），但本文每步使用固定预算，留作未来工作
5. **分组策略固定**：组划分为规则网格，未探索基于内容的自适应分组策略
6. **Read/Write 层增加参数量**：DiT-XL 从 675M 增至 698M（+3.4%），对超大模型可能不可忽略
7. **与更先进的 token 压缩方法未对比**：如 Token Merging、SparseDiT 等方法可能与 ELIT 互补但未实验验证

## 与相关工作的对比

| 方法类别 | 代表工作 | 与 ELIT 的区别 |
|----------|----------|----------------|
| 自适应生成器 | FlexiDiT, Supernetwork | 均匀分配计算 / 路由器复杂度高；ELIT 通过潜变量接口非均匀分配 |
| Token 丢弃加速训练 | MaskDiT, MDTv2, TREAD | 推理时必须恢复全 token，无法加速推理；ELIT 推理时可变 |
| 训练无关加速 | Token Merging, SparseDiT | 不改善训练效率；ELIT 训练也加速 3-4× |
| 潜变量接口 | RIN, FIT | 偏离 DiT 架构，需特殊优化器；ELIT 即插即用 |
| 可变长度 autoencoder | FlexTok, TiTok | 弹性止步于编码器；ELIT 将弹性引入生成模型内部 |

ELIT 的核心优势在于「最小改动、最大兼容」——不改变训练目标、不引入辅助损失、不需要特殊优化器，仅通过两个跨注意力层和尾部丢弃训练策略就实现了推理弹性和自适应计算分配，这使其相比上述所有方法都更易于集成到现有 DiT 生态系统中。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将可变长度潜变量接口以极简方式融入 DiT 生态，tail dropping 形成重要性排序的设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 四种架构 × 两种分辨率 × 视频 × 20B 大模型 × 详尽消融，覆盖面极广
- 写作质量: ⭐⭐⭐⭐ — 动机清晰（零填充实验很有说服力），结构清楚，图表丰富
- 价值: ⭐⭐⭐⭐⭐ — 解决了 DiT 推理效率的核心痛点，设计足够简单可被广泛采用，多预算推理有很强的实用价值
- 综合: ⭐⭐⭐⭐⭐ — Snap Research + Rice 的扎实工作，兼具理论洞见（均匀计算浪费）和工程价值（即插即用加速），CVPR oral 级别

> **推荐阅读**：Section 3.2 的零填充实验和 Figure 2 的注意力可视化是全文最精彩的部分，清晰展示了 DiT 的计算浪费问题和 ELIT 的解决思路。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)
- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
