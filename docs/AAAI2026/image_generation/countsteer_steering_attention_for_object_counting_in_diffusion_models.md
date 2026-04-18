---
title: >-
  [论文解读] CountSteer: Steering Attention for Object Counting in Diffusion Models
description: >-
  [AAAI2026][图像生成][扩散模型] 提出 CountSteer，一种免训练的推理时方法，通过在扩散模型的 cross-attention 隐状态中注入自适应 steering vector，将物体计数准确率提升约 4%，且不损害图像质量。
tags:
  - AAAI2026
  - 图像生成
  - 扩散模型
  - object counting
  - steering vector
  - 注意力机制
  - training-free
---

# CountSteer: Steering Attention for Object Counting in Diffusion Models

**会议**: AAAI2026  
**arXiv**: [2511.11253](https://arxiv.org/abs/2511.11253)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: diffusion models, object counting, steering vector, cross-attention, training-free

## 一句话总结

提出 CountSteer，一种免训练的推理时方法，通过在扩散模型的 cross-attention 隐状态中注入自适应 steering vector，将物体计数准确率提升约 4%，且不损害图像质量。

## 背景与动机

- 文本到图像 (T2I) 扩散模型（如 Stable Diffusion）已能生成高度逼真的图像，但在遵循文本中的数量指令方面仍然表现不佳，经常出现物体数量偏多或偏少的问题
- 物体计数是衡量生成保真度（fidelity）的一个清晰且可量化的指标，但模型在随机去噪过程中难以稳定保持数量信息
- 现有改进方案主要依赖微调（LoRA、DreamBooth）或结构修改（Composer、T2I-Adapter），增加了训练和部署成本
- 作者通过核密度估计（KDE）分析发现了一个关键直觉：**扩散模型并非完全"数盲"，其内部 cross-attention 的隐状态分布在计数正确与错误的样本之间存在可分离的方向性差异**，说明模型已经隐式编码了数量相关信息，只是未能稳定表达

## 核心问题

如何在不重新训练、不修改模型结构的前提下，利用扩散模型内部已有的数量感知信号，在推理时引导生成过程，使输出图像中的物体数量与文本指定数量一致？

## 方法详解

### 1. 核心思路：Steering Vector

借鉴大语言模型中的 Inference-Time Intervention (ITI) 思想。ITI 发现 LLM 的隐表示在"真实"和"不真实"类别之间存在线性可分方向，可通过注入方向向量控制输出。CountSteer 将此概念迁移到扩散模型的 UNet 结构中：

- 对每个去噪步骤 $t$ 和 UNet block $b$，分别计算计数正确样本（Class 1）和错误样本（Class 0）的隐状态均值 $\mu_{t,b}^1$ 和 $\mu_{t,b}^0$
- 基础 steering vector 定义为二者之差：$s_{t,b} = \mu_{t,b}^1 - \mu_{t,b}^0$

### 2. 数据集构建

- 使用 GPT-4o 自动生成 600 个 "{count} {object}" 格式的 prompt（如 "three cats"），数量范围限制在 1-4（因为超过 4 后模型几乎完全失败）
- 400 个用于构建 steering vector，200 个用于评估，无重叠
- 每个 prompt 生成一张图像，人工标注为 Class 0 或 Class 1；通过更换随机种子重新生成来确保类别平衡
- 从标注图像中提取前 $k$ 个去噪步的 cross-attention query 向量作为隐状态

### 3. 自适应缩放机制

固定 steering vector 在不同 prompt 下效果不一（有时矫正不足，有时过度矫正）。CountSteer 引入自适应缩放因子 $\alpha_{t,b}$，包含两个关键度量：

**距离缩放**：计算当前隐状态 $h_{t,b}$ 到目标分布均值的距离比：

$$d_{t,b} = \frac{\|\delta_{t,b}\|_2}{\|s_{t,b}\|_2}, \quad \delta_{t,b} = \mu_{t,b}^1 - h_{t,b}$$

**方向对齐**：通过余弦相似度判断 steering vector 是否指向正确方向。

**组合缩放因子**：

$$\alpha_{t,b} = \cos(s_{t,b}, \delta_{t,b}) \cdot (1 - e^{-d_{t,b}}) \cdot c$$

- 当隐状态远离目标（$d_{t,b} > 1$）时指数项趋近 1，允许更强修正
- 当隐状态接近目标（$d_{t,b} < 1$）时修正自动减弱，防止过度矫正
- $c = 100$ 为全局放大常数（因余弦和指数项的乘积值本身较小）

### 4. 推理时注入

在每个去噪步骤 $t$，将自适应 steering vector 注入 UNet 各 block 的隐状态：

$$h_{t,b}' = h_{t,b} + \alpha_{t,b} \cdot s_{t,b}$$

仅在前 10 个去噪步（$k=10$）注入，因为已有研究表明全局布局和粗结构主要在早期去噪阶段确定。

## 实验关键数据

**骨干模型**：Stable Diffusion v1.5，50 步去噪，guidance scale 7.5

**评估方式**：使用 LLaVA-OneVision 自动计数生成图像中的物体数量

| 方法 | ACC ↑ | MAE ↓ | CLIP-Score ↑ |
|------|-------|-------|-------------|
| SD v1.5 (Baseline) | 50.0% | 1.125 | 30.99 |
| SD v1.5 + CountSteer | **54.0%** | **0.940** | 30.39 |

- 准确率提升 4.0 个百分点
- MAE 降低 0.185，显著减少极端偏差
- CLIP-Score 基本持平，说明语义对齐和图像质量未受损

## 亮点

1. **发现了扩散模型内部的数量感知信号**：通过 KDE 分析揭示 cross-attention 隐状态在计数正确/错误样本之间线性可分，这一观察本身具有分析价值
2. **完全免训练**：无需微调、无需修改模型结构，仅在推理时注入向量，部署成本极低
3. **自适应机制设计合理**：距离缩放 + 方向对齐 + 指数衰减的组合使得修正强度能根据当前状态动态调整，避免了固定 steering vector 的不稳定性
4. **概念清晰、方法简洁**：从 LLM 领域的 ITI 迁移到扩散模型，跨领域思路有启发性

## 局限与展望

1. **数量范围有限**：仅支持 1-4 的计数范围，超过 4 时模型本身几乎完全失败，steering 也无法拯救
2. **提升幅度偏小**：4% 的准确率提升虽然一致，但绝对值不大，50%→54% 仍意味着近一半图像计数错误
3. **仅在 SD v1.5 上验证**：缺少在 SDXL、Flux 等更新模型上的实验，泛化性存疑
4. **失败模式明显**：存在过度生成、渲染失败、以及原本正确被 steering 反而搞错三类失败情况
5. **数据集构建需要人工标注**：构建 steering vector 的 400 个样本需要手动标注计数正确性，自动化程度不高
6. **缺少与同期工作的对比**：未与 Attend-and-Excite、Divide-and-Bind 等其他免训练物体计数方法做直接对比

## 与相关工作的对比

| 方法类型 | 代表工作 | 与 CountSteer 区别 |
|---------|---------|-------------------|
| 微调方法 | LoRA, DreamBooth | 需要额外训练，CountSteer 免训练 |
| 结构修改 | Composer, T2I-Adapter | 需要修改网络结构，CountSteer 不改模型 |
| LLM Steering | ITI (Li et al., 2023) | 作用于语言模型注意力头，CountSteer 迁移到扩散模型 UNet |
| 注意力操控 | Attend-and-Excite | 操控 attention map 以增强忽略 token，CountSteer 操控隐状态方向 |

## 启发与关联

- **Steering 范式的广泛适用性**：从 LLM 的真实性控制到扩散模型的计数控制，steering vector 思路展现了跨模态的可迁移性。未来可将同样的框架扩展到颜色一致性、空间布局、多物体交互等组合属性
- **扩散模型的"隐式知识"挖掘**：模型知道什么是正确的但做不到——这一发现提示我们，与其训练新能力，不如挖掘和激活已有能力，这可能是一个更高效的研究范式
- **与 Classifier-Free Guidance 的关系**：CFG 通过条件/无条件方向引导生成，CountSteer 通过正确/错误方向引导计数，两者在几何层面上有结构相似性，值得进一步统一
- **实用性思考**：当前 4% 的提升幅度在应用层面价值有限，但如果与 layout guidance 等方法结合使用，可能产生协同效果

## 评分

- 新颖性: 3/5（LLM steering 到扩散模型的迁移有新意，但方法本身较为直觉化）
- 实验充分度: 2/5（仅 SD v1.5，缺少 baseline 对比，数量范围小）
- 写作质量: 3/5（动机清晰、结构完整，但实验部分偏薄）
- 价值: 3/5（提供了有价值的分析发现，但实际提升有限）
