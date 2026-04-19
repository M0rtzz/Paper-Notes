---
title: >-
  [论文解读] VORTA: Efficient Video Diffusion via Routing Sparse Attention
description: >-
  [NeurIPS 2025][视频理解][视频扩散模型加速] 提出VORTA框架，通过桶化核心集注意力（建模长程依赖）和信号感知路由机制（自适应选择稀疏注意力分支），在不损失生成质量的前提下实现视频扩散Transformer端到端1.76×加速，并可与缓存和蒸馏方法叠加达到14.41×加速。
tags:
  - NeurIPS 2025
  - 视频理解
  - 视频扩散模型加速
  - 稀疏注意力
  - 路由机制
  - 核心集选择
  - 视频生成
---

# VORTA: Efficient Video Diffusion via Routing Sparse Attention

**会议**: NeurIPS 2025  
**arXiv**: [2505.18809](https://arxiv.org/abs/2505.18809)  
**代码**: [GitHub](https://github.com/wenhao728/VORTA)  
**领域**: 视频理解  
**关键词**: 视频扩散模型加速, 稀疏注意力, 路由机制, 核心集选择, 视频生成

## 一句话总结

提出VORTA框架，通过桶化核心集注意力（建模长程依赖）和信号感知路由机制（自适应选择稀疏注意力分支），在不损失生成质量的前提下实现视频扩散Transformer端到端1.76×加速，并可与缓存和蒸馏方法叠加达到14.41×加速。

## 研究背景与动机

### 视频扩散Transformer的效率瓶颈

视频扩散Transformer（VDiT）在高质量视频生成方面取得了显著进展，但计算代价极高。以HunyuanVideo为例，生成5秒720p视频需要约1000秒（500 PFLOPS），其中注意力操作占计算量的75%以上。

3D自注意力的复杂度为 $\mathcal{O}(L^2 d)$，其中序列长度 $L = F \times H \times W$。即使经过VAE和patchification压缩，HunyuanVideo的序列长度仍高达约10万token。

### 现有加速方法的局限

**局部稀疏方法**（如STA）：利用注意力分数集中于局部邻域的特性限制交互范围，但对长程注意力头效果差。最近的4%的key贡献了超过80%的注意力权重，但剩余96%的key在早期采样步骤中仍然重要。

**在线分析方法**（如ARnR）：动态检测稀疏模式，但引入 $\mathcal{O}(L^2)$ 的相似度计算开销，且在采样配置变更时需重新调参。

**核心矛盾**：VDiT中同时存在局部注意力和长程注意力，且两者在采样过程中会动态切换，简单的静态策略无法兼顾。

### VDiT注意力的三种分类

作者将VDiT中的注意力分为三类：
- **局部注意力**：关注短程交互，负责精细细节
- **长程注意力**：分布于整个序列，捕获高层语义（布局和运动），对微小扰动（如合并相似token）容忍度高
- **关键注意力**：同时维护全局感知和局部细节，对扰动非常敏感

关键发现：长程注意力头具有高序列内冗余性——token高度相似，少量代表性token就能概括其信息。

## 方法详解

### 整体框架

VORTA包含两个核心组件：(1) 针对不同注意力类型的稀疏注意力变体；(2) 基于信噪比的路由机制，自适应为每个注意力头选择最优稀疏策略。

### 关键设计

#### 1. 滑动窗口注意力（针对局部注意力）

采用3D滑动瓦片注意力（Sliding Tile Attention），将锯齿形注意力掩码转化为块状密集掩码以提高GPU硬件效率：

$$\mathbf{M} = \{m_{i,j}\} = \{j \in (\tau(i)-w, \tau(i)+w]\}$$

其中 $\tau(i) = \lfloor i/t \rfloor \cdot t + \lceil t/2 \rceil$ 为瓦片中心。窗口大小设为 $(18, 27, 24)$。

#### 2. 桶化核心集注意力（针对长程注意力）

**核心思路**：对长程注意力头，先通过核心集选择剪枝冗余token，然后在压缩后的序列上执行注意力。

$$\text{coreset-attn}(\mathbf{H}) = \text{unpool} \circ \text{attn} \circ \text{pool}(\mathbf{H})$$

**桶化核心集选择（BCS）**：将token划分为大小 $(t, h, w) = (2, 3, 2)$ 的桶，每个桶内计算中心token与邻居的相似度，剪枝相似度最高的top-k个token（核心集比率 $r_{\text{core}} = 0.5$）。

**复杂度优势**：BCS仅需 $\mathcal{O}(L)$ 复杂度（每个桶 $\mathcal{O}(thw)$ × $L/(thw)$ 个桶），而非全局配对方法的 $\mathcal{O}(L^2)$。核心集保留50%的token，注意力计算减少至原来的25%（二次方关系）。

相比标准平均池化，BCS的关键优势在于选择性：当相邻token差异大时，简单平均会造成信息损失导致马赛克或模糊，而BCS通过剪枝最相似的token保留多样性。

#### 3. 信号感知注意力路由

**设计动机**：注意力行为与输入feature的信噪比强相关——早期步骤长程注意力更多（构建高层语义），后期步骤局部注意力更多（精化细节）。

路由器为每层一个线性层，以扩散时间步嵌入 $\mathbf{T}$ 为输入：

$$\boldsymbol{\alpha}^{(n)} = \text{softmax}(\mathbf{T} \mathbf{W}_R^{(n)})$$

推理时硬选择门控值最大的分支：

$$\mathbf{H}^{(n+1)} = \begin{cases} \text{sliding-attn}(\mathbf{H}^{(n)}) & \text{if } \alpha_2 > \alpha_1, \alpha_3 \\ \text{coreset-attn}(\mathbf{H}^{(n)}) & \text{if } \alpha_3 > \alpha_1, \alpha_2 \\ \text{attn}(\mathbf{H}^{(n)}) & \text{otherwise} \end{cases}$$

仅增加0.1%的参数量，且不引入推理时开销。实际运行中，全注意力分支仅在约0.2%的情况下被选中。

### 损失函数 / 训练策略

路由器训练使用自监督蒸馏策略，冻结VDiT原始参数，仅更新路由器权重：

$$\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{distill}} \cdot \text{MSE}(\mathbf{H}_{\text{org}}^{(N)}, \mathbf{H}^{(N)}) + \lambda_{\text{reg}} \cdot \sum_{n=1}^{N} \|\alpha_1^{(n)}\|^2$$

其中 $\lambda_{\text{distill}} = 20$，$\lambda_{\text{reg}} = 0.02$。L2正则化促进稀疏选择。训练仅需100步（Mixkit数据集），2块H100 GPU约一天完成。

## 实验关键数据

### HunyuanVideo主实验

| 方法 | 类型 | VBench↑ | LPIPS↓ | 延迟(s) | 加速比 | 显存(GB) |
|------|------|---------|--------|---------|--------|----------|
| HunyuanVideo | - | 82.26 | - | 1043.85 | 1.00× | 47.64 |
| + ARnR | 稀疏 | 82.39 | 0.211 | 790.55 | 1.32× | 78.15 |
| + STA | 稀疏 | 82.33 | 0.201 | 676.39 | 1.54× | 51.79 |
| + PAB | 缓存 | 82.40 | 0.186 | 815.51 | 1.28× | >80 |
| **+ VORTA** | **稀疏** | **82.59** | **0.185** | **594.23** | **1.76×** | **51.15** |
| + VORTA & PAB | 组合 | 82.56 | 0.195 | 444.19 | 2.35× | >80 |
| + PCD | 蒸馏 | 81.17 | 0.564 | 125.98 | 8.29× | 47.64 |
| + VORTA & PCD | 组合 | 81.49 | 0.575 | 72.46 | **14.41×** | 51.15 |

### 消融实验（Wan 2.1 1.3B, 480p）

| 配置 | VBench↑ | 延迟(s) | 加速比 |
|------|---------|---------|--------|
| Wan 2.1基线 | 81.20 | 73.24 | 1.00× |
| w/o 滑动注意力 | 80.25 | 65.14 | 1.12× |
| w/o 核心集注意力 | 79.89 | 66.10 | 1.11× |
| w/o 全注意力（去掉保底分支） | 77.14 | 59.34 | 1.23× |
| w/o 时间步条件 | 81.03 | 65.00 | 1.13× |
| 平均池化 AP(2,1,1) | 77.08 | 57.53 | 1.27× |
| 平均池化 AP(1,2,1) | 76.01 | 57.64 | 1.27× |
| **VORTA完整** | **81.06** | **58.42** | **1.25×** |

### 关键发现

1. VORTA在VBench上甚至略超原始模型（82.59 vs 82.26），可能因剪枝注意力冗余带来的正则化效果
2. 去掉全注意力分支导致VBench下降4分但无额外加速，证明关键注意力的存在
3. 去掉时间步条件使路由器在所有步骤中统一选择相同分支，性能下降且加速减少
4. BCS大幅优于简单平均池化（81.06 vs 75.94-77.08），验证了选择性剪枝的必要性
5. 路由模式呈现清晰的时间规律：早期以核心集注意力为主，后期转向滑动注意力

## 亮点与洞察

1. **理论设计严谨**：三类注意力分类为不同稀疏策略提供了清晰的理论基础
2. **BCS的线性复杂度**：通过桶化策略将核心集选择从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(L)$，是实用加速的关键
3. **强兼容性**：与缓存（PAB）和蒸馏（PCD）方法正交，叠加后达到14.41×加速
4. **调度器泛化**：无需重新分析即可适配不同的ODE求解器和步数设置
5. **模型骨干泛化**：在MMDiT（HunyuanVideo）和DiT（Wan 2.1）上均验证有效

## 局限与展望

- 主要针对注意力机制加速，对短序列（图像或低分辨率视频）的加速空间有限
- 仅支持双向生成范式，自回归视频生成需要较大适配
- 预训练模型本身质量差时（生成变形或违反物理），VORTA会继承甚至放大这些问题
- 路由器需要约一天训练，虽然远小于预训练成本但仍非零成本
- 核心集比率固定为50%，自适应比率可能进一步优化效率-质量权衡

## 相关工作与启发

- **稀疏注意力**: STA (滑动瓦片注意力), ARnR (在线分析), SVG
- **视频扩散加速**: PAB (特征缓存), PCD (一致性蒸馏)
- **条件计算**: MoE (混合专家), BlockDrop
- **视频生成**: HunyuanVideo, Wan 2.1, CogVideoX
- **启发**: 路由机制的设计思路可推广到其他条件计算场景；BCS的线性复杂度核心集选择可用于LLM长文本推理

## 评分
- 新颖性: ⭐⭐⭐⭐☆ — 核心集注意力和信号感知路由的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 多骨干、多调度器、多组合的全面评估+详细运行时分析
- 写作质量: ⭐⭐⭐⭐⭐ — 分类学清晰，图表设计出色
- 价值: ⭐⭐⭐⭐⭐ — 实际加速效果显著，已开源代码和权重

<!-- RELATED:START -->

## 相关论文

- [Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [VSA: Faster Video Diffusion with Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation](s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)
- [LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)
- [Training-Free Efficient Video Generation via Dynamic Token Carving (Jenga)](training-free_efficient_video_generation_via_dynamic_token_carving.md)

<!-- RELATED:END -->
