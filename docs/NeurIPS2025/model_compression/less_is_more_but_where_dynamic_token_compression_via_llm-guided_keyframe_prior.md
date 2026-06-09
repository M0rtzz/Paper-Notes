---
title: >-
  [论文解读] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior
description: >-
  [NeurIPS 2025][模型压缩][video-understanding] 提出 DyToK，一种无需训练的视频 token 动态压缩方法，利用 VLLM 深层注意力中固有的 query 条件关键帧先验，为不同帧自适应分配 token 预算，实现即插即用式的效率-精度最优权衡。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "video-understanding"
  - "token-compression"
  - "vllm"
  - "efficiency"
  - "keyframe-selection"
---

# Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior

**会议**: NeurIPS 2025  
**arXiv**: [2512.06866](https://arxiv.org/abs/2512.06866)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: video-understanding, token-compression, vllm, efficiency, keyframe-selection  

## 一句话总结

提出 DyToK，一种无需训练的视频 token 动态压缩方法，利用 VLLM 深层注意力中固有的 query 条件关键帧先验，为不同帧自适应分配 token 预算，实现即插即用式的效率-精度最优权衡。

## 背景与动机

1. **长视频计算瓶颈**：Video LLM 的计算复杂度与视觉 token 序列长度呈二次增长，长视频推理效率极低，限制了实际部署。
2. **均匀压缩不合理**：现有编码器特征方法（VisionZip、LLaMA-VID）对所有帧施加统一压缩比，忽略了不同帧的时序重要性差异——关键帧被过度裁剪，冗余帧保留过多。
3. **LLM 注意力剪枝不稳定**：FastV 等基于 LLM 注意力的方法依赖特定中间层的注意力图，浅层噪声导致剪枝质量不稳定，深层又丧失效率优势。
4. **关键帧选择是二元决策**：已有关键帧选择方法（AKS 等）将帧分为"保留/丢弃"的二元决策，丢弃帧中潜在有用信息被不可逆丢失，保留帧中冗余信息未被清理。
5. **额外计算开销**：现有关键帧选择依赖预训练 VLM 或辅助模块在特征编码前进行选择，引入额外计算成本，反而削弱效率收益。
6. **VLM 注意力中隐含的关键帧先验**：实验发现 VLLM 注意力层天然会对 query 相关的关键帧赋予更高注意力权重，且即使回答错误，注意力仍能正确定位关键帧，但这一先验未被充分利用。

## 方法详解

### 3.1 整体框架

DyToK 是一个两阶段 training-free 框架：
1. **时序重要性估计**：利用轻量辅助模型的深层注意力获取帧级重要性分数
2. **动态帧级压缩**：根据重要性分数按比例分配每帧 token 预算，再调用任意兼容的 token 剪枝方法执行压缩

### 3.2 时序重要性估计

计算最后一个文本 query token 与各帧视觉 token 之间的交叉模态注意力分数：

$$w_f = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \text{Softmax}\left(\frac{\mathbf{Q}_l \mathbf{K}_l^\top}{\sqrt{D}}\right)$$

**关键发现**：深层注意力提供显著更好的关键帧先验（第 20 层最佳，见消融实验）。

**轻量辅助模型**：直接用主模型的深层注意力引导浅层需重新计算，计算开销翻倍。解决方案是使用同架构族的 0.5B 小模型作为辅助，仅 1/14 计算量即可达到近似的关键帧识别精度。

### 3.3 动态帧级压缩

核心是 token 预算分配算法：
1. **初始分配**：$a_f = \lfloor \hat{w}_f \times T_{\text{total}} \rfloor$
2. **余量分配**：计算各帧小数余量 $r_f$，按余量降序分配剩余 token
3. **上限截断**：超过每帧上限 $T_{\max}$ 的 token 按重要性排名重分配给未满帧
4. **模块化压缩**：最终调用 $\text{Compression}(x_f, a_f)$ 执行实际剪枝，可即插即用替换为 VisionZip/FastV/DyCoke 等任意兼容方法

### 3.4 设计亮点

- 突破二元选择范式：不再是"选帧/弃帧"，而是"多保留/少保留"
- 无需训练：完全利用预训练模型的内在注意力机制
- 即插即用：同时兼容编码器特征方法和 LLM 注意力方法

## 实验结果

### 表1：编码器特征方法上的增强效果（LLaVA-OneVision，32帧）

| 方法 | 保留Token数 | 压缩率 | VideoMME | LongVideoBench | MLVU | 平均 | 相对% |
|------|-----------|--------|----------|---------------|------|------|-------|
| Vanilla | 6272 | — | 58.5 | 56.6 | 47.1 | 54.1 | 100 |
| VisionZip | 3136 | ↓50% | 57.2 | 53.8 | 43.7 | 51.6 | 95.4 |
| VisionZip†+DyToK | 3136 | ↓50% | 59.1 | 56.4 | 46.2 | **53.9** | **99.6** |
| VisionZip | 448 | ↓90% | 44.5 | 41.4 | 29.8 | 38.6 | 71.3 |
| VisionZip†+DyToK | 448 | ↓90% | 53.2 | 50.4 | 42.8 | **48.8** | **90.2** |

**发现**：50% 压缩下 DyToK 提升 4.2%，90% 极端压缩下提升 18.9%，压缩率越高收益越大。

### 表2：LLM 注意力方法上的增强效果

| 方法 | 保留Token数 | 压缩率 | VideoMME | LongVideoBench | MLVU | 平均 | 相对% |
|------|-----------|--------|----------|---------------|------|------|-------|
| FastV | 4704 | ↓25% | 57.6 | 57.1 | 46.5 | 53.7 | 99.3 |
| FastV+DyToK | 4704 | ↓25% | 58.4 | 56.8 | 46.8 | **54.0** | **99.8** |
| FastV | 896 | ↓85% | 51.1 | 51.2 | 38.3 | 46.9 | 86.7 |
| FastV+DyToK | 896 | ↓85% | 54.8 | 52.6 | 43.2 | **50.2** | **92.8** |

**发现**：在 LLM 注意力方法上，85% 压缩下 DyToK 提升 FastV 6.1%，证明跨范式兼容性。

### 消融实验关键结论

- **注意力层位置**：第 20 层（共 24 层）提供最佳关键帧先验，浅层（0-8层）效果差，验证"深层注意力编码高级语义"假设
- **辅助模型大小**：0.5B 模型与 7B 模型的关键帧先验质量几乎一致（性能差距 ≤1.5%），而计算开销仅为 1/14

## 亮点

- 发现并验证了 VLLM 深层注意力中固有的 query 条件关键帧先验：即使模型回答错误，注意力仍能正确定位关键帧
- 从"二元帧选择"到"连续 token 预算分配"的范式创新，更精细地平衡效率与信息保留
- 在 90% 极端压缩率下仍保持未压缩模型 90.2% 的精度，且可达 4.3× 加速
- 零训练、即插即用设计，同时兼容两大类压缩方法，实际应用门槛极低

## 局限性

- 仍需引入额外的轻量辅助模型（0.5B）来提取关键帧先验，增加了系统复杂性，未能完全避免额外模型开销
- 仅在 LLaVA-OneVision 和 Qwen2.5-VL 等少数架构上验证，对其他视频 LLM（如 VideoChat、Video-LLaMA）的泛化性未知
- 关键帧先验的质量取决于辅助模型与主模型的架构同族性，跨架构族的迁移可行性未探讨
- 主要在选择题式 benchmark（VideoMME、LongVideoBench、MLVU）上评估，开放式生成任务的效果未验证
- 对于视觉内容变化极小的视频（如监控画面），按帧重要性分配 token 的收益可能有限

## 相关工作对比

### vs FastV (ECCV 2024)
FastV 在 LLM 推理过程中基于特定中间层的注意力图动态剪枝视觉 token，但其效果严重依赖选择的层——浅层注意力噪声大导致剪枝不稳定。DyToK 不修改推理过程本身，而是在推理前通过辅助模型的深层注意力预分配帧级 token 预算，与 FastV 正交互补。85% 压缩下 DyToK 使 FastV 提升 6.1%。

### vs VisionZip (CVPR 2025)
VisionZip 利用编码器特征的 patch 间相关性静态选择 token，但对所有帧施加统一压缩比，忽略时序动态。此外原版 VisionZip 丢弃空间位置信息，与主流 2D pooling 方法不兼容。DyToK 引入帧级动态压缩比，与改进版 VisionZip† 结合后在 90% 压缩下提升 18.9%。

### vs 关键帧选择方法 (AKS 等)
AKS 等方法使用预训练 VLM 在特征编码前做二元帧选择，引入额外计算开销且丢弃帧中的潜在有用信息。DyToK 采用连续 token 预算分配替代二元决策，即使对"不重要"帧也保留少量 token，避免信息的不可逆丢失。

## 评分

- ⭐⭐⭐⭐ 新颖性：从二元帧选择到连续 token 预算分配的范式转变
- ⭐⭐⭐⭐ 技术质量：关键帧先验发现有理论深度，消融实验充分
- ⭐⭐⭐⭐⭐ 实验充分度：三大 benchmark × 两大类方法 × 多压缩率的全面评估
- ⭐⭐⭐⭐ 可复现性：training-free，代码已开源，易于集成

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference](kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)
- [\[ACL 2025\] LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](../../ACL2025/model_compression/llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)
- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)

</div>

<!-- RELATED:END -->
