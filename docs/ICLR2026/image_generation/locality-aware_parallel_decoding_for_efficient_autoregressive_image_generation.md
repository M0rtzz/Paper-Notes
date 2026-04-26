---
title: >-
  [论文解读] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation
description: >-
  [ICLR 2026][图像生成][并行解码] 提出 Locality-aware Parallel Decoding (LPD)，通过灵活并行化自回归建模架构和局部性感知的生成顺序调度，将 256×256 图像的生成步数从 256 降至 20，实现至少 3.4× 的延迟降低。
tags:
  - ICLR 2026
  - 图像生成
  - 并行解码
  - 自回归建模
  - 空间局部性
  - 位置查询
  - 高效推理
---

# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

**会议**: ICLR 2026  
**arXiv**: [2507.01957](https://arxiv.org/abs/2507.01957)  
**代码**: [GitHub](https://github.com/mit-han-lab/lpd)  
**领域**: 自回归图像生成  
**关键词**: 并行解码, 自回归建模, 空间局部性, 位置查询, 高效推理

## 一句话总结
提出 Locality-aware Parallel Decoding (LPD)，通过灵活并行化自回归建模架构和局部性感知的生成顺序调度，将 256×256 图像的生成步数从 256 降至 20，实现至少 3.4× 的延迟降低。

## 研究背景与动机
- 自回归图像生成的 next-patch prediction 是内存瓶颈操作，延迟随步数线性增长
- next-scale prediction（如 VAR）步数少但使用多尺度token表示，与平坦视觉感知模型（CLIP、DINO）不兼容
- 现有并行化方法（PAR、RandAR）仅实现有限并行化，PAR 固定并行顺序，RandAR 并行token之间互不可见
- 需要：高效推理 + 保持平坦token表示的通用性和兼容性

## 方法详解

### 整体框架
LPD 包含两个核心：灵活并行化自回归建模架构（支持任意生成顺序和并行度）+ 局部性感知生成顺序调度（最大化上下文支持，最小化组内依赖）。

### 关键设计

1. **灵活并行化自回归建模**: 解耦上下文表示和token生成——已生成token提供上下文（KV cache），可学习位置查询token驱动目标位置的并行生成。使用专门的注意力掩码：

    - Context Attention：因果地让后续token关注上下文token
    - Query Attention：同一步的位置查询token相互可见，但不允许后续token关注查询token
   推理时编码和解码可融合为单步操作，仅存储生成token的KV cache。

2. **局部性分析**: 在 LlamaGen-1.4B 上分析注意力模式，发现强空间局部性——解码token的注意力集中在附近空间token上。定义 Per-Token Attention (PTA)：
$$PTA_s = \frac{1}{N}\sum_{i=1}^N \frac{\sum_j \text{Attention}(T_i,T_j) \cdot \mathbb{I}[d(T_i,T_j)=s]}{\sum_j \mathbb{I}[d(T_i,T_j)=s]}$$
   PTA 随距离急剧下降，验证了两个原则：并行token应靠近已生成token（强条件化）、远离同组token（低依赖）。

3. **局部性感知生成顺序调度**: 在每步 $k$ 中：

    - 计算未选token与已选token的欧氏距离作为proximity
    - 按proximity排序，阈值 $\tau$ 筛选高proximity候选集 $c_1$
    - 从 $c_1$ 中依次选取token，每选一个就用排斥阈值 $\rho$ 过滤邻近token
    - 不足时用最远点采样从剩余集 $c_2$ 补充
   组大小通常通过余弦调度递增。生成顺序可预计算。

### 损失函数 / 训练策略
分组自回归训练目标：$p(x_1,...,x_N;c) = \prod_{g=1}^G p(X_g|X_{<g};c)$
使用交叉熵损失，训练时采用专门的注意力掩码实现 teacher-forcing + 并行预测。

## 实验关键数据

### 主实验（ImageNet 256×256）

| 类型 | 模型 | 参数 | FID↓ | IS↑ | #Steps | Latency(s) | Throughput |
|------|------|------|------|-----|--------|-----------|-----------|
| AR | LlamaGen-XXL | 1.4B | 2.34 | 253.9 | 576 | 24.40 | 0.72 |
| AR | RAR-XXL | 1.5B | 1.48 | 326.0 | 256 | 6.59 | 6.72 |
| Par.AR | PAR-XXL-4× | 1.4B | 2.35 | 263.2 | 147 | 6.26 | 2.33 |
| Par.AR | RandAR-L | 343M | 2.55 | 288.8 | 88 | 1.97 | 28.59 |
| **Par.AR** | **LPD-L** | **343M** | **2.31** | **284.9** | **20** | **0.40** | **92.42** |
| **Par.AR** | **LPD-XL** | **775M** | **1.97** | **304.0** | **20** | **0.57** | **60.27** |

### ImageNet 512×512

| 模型 | 参数 | FID↓ | #Steps | Latency(s) | Throughput |
|------|------|------|--------|-----------|-----------|
| LlamaGen-XXL | 1.4B | 2.59 | 1024 | - | - |
| **LPD-XXL** | **1.4B** | **2.25** | **48** | **2.78** | **6.56** |

### 关键发现
- LPD-L 仅 20 步生成 256×256 图像，FID=2.31 优于 576 步的 LlamaGen-XXL (2.34)
- 吞吐量 92.42 img/s 远超 RandAR 的 28.59 和 PAR 的 6.83
- 512×512 仅需 48 步（vs 1024），FID 从 2.59 降至 2.25
- 局部性感知调度远优于光栅序、随机序和 Halton 序
- 零样本图像编辑（类条件编辑、修复、扩展）自然支持

## 亮点与洞察
- 位置查询token实现的"解耦"设计优雅地解决了标准decoder-only模型的灵活性限制
- Query Attention 确保同步生成token之间互相可见，避免独立采样导致的不一致
- 局部性分析提供了并行化策略设计的经验基础——PTA 分析可迁移到其他视觉自回归模型
- 与 VAR 相比保持了平坦token表示，兼容 CLIP/DINO 等视觉骨干

## 局限与展望
- 当前仅在 ImageNet 类条件生成上验证，未扩展到文本引导生成
- 位置查询token引入的额外参数和注意力计算的开销
- 生成顺序调度的超参（$\tau$、$\rho$、组大小调度）需要调优
- 与最佳 MAR/VAR 方法在 FID 上仍有差距（但吞吐量远优）

## 相关工作与启发
- PAR、RandAR、SAR 等并行自回归方法的局限驱动了本工作
- MaskGIT 的掩码预测启发了渐增组大小的设计
- 空间局部性观察对理解视觉自回归模型的注意力机制有启发
- 为统一多模态生成（文本+图像）中的图像部分提供了高效解决方案

## 技术细节补充
- 组大小通过余弦调度递增：早期上下文少时生成少量 token，后期增多
- 位置查询 token = 共享可学习嵌入 + 目标位置的位置编码
- 推理时查询 token 的 KV 不存储，仅存储生成 token 的 KV
- 256×256 生成 20 步，512×512 生成 48 步
- 支持零样本图像编辑（类条件编辑、修复、扩展）
- LPD-L 343M 参数即可达到 FID=2.31，超越 1.4B 的 LlamaGen-XXL

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 位置查询解耦+局部性感知调度的组合设计新颖有效
- 实验充分度: ⭐⭐⭐⭐ 系统对比充分，但缺少T2I和多模态实验
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法与其他方法的对比分析透彻
- 价值: ⭐⭐⭐⭐⭐ 大幅降低自回归图像生成延迟，对统一多模态系统有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)
- [\[ICLR 2026\] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)
- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)

<!-- RELATED:END -->
