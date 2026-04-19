---
title: >-
  [论文解读] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images
description: >-
  [NeurIPS 2025][图像生成][水印伪造攻击] 提出 WMCopier，首个基于扩散模型的 no-box 水印伪造攻击方法，无需任何目标水印算法的先验知识，通过训练无条件扩散模型学习水印分布、浅层反演注入水印信号、迭代精炼优化质量，在开源和商业水印系统（包括 Amazon）上实现高成功率伪造。
tags:
  - NeurIPS 2025
  - 图像生成
  - 水印伪造攻击
  - 扩散模型
  - DDIM 反演
  - 不可见水印
  - No-Box 攻击
---

# WMCopier: Forging Invisible Image Watermarks on Arbitrary Images

**会议**: NeurIPS 2025  
**arXiv**: [2503.22330](https://arxiv.org/abs/2503.22330)  
**代码**: [GitHub](https://github.com/holdrain/WMCopier)  
**领域**: 图像水印安全 / 扩散模型  
**关键词**: 水印伪造攻击, 扩散模型, DDIM 反演, 不可见水印, No-Box 攻击

## 一句话总结

提出 WMCopier，首个基于扩散模型的 no-box 水印伪造攻击方法，无需任何目标水印算法的先验知识，通过训练无条件扩散模型学习水印分布、浅层反演注入水印信号、迭代精炼优化质量，在开源和商业水印系统（包括 Amazon）上实现高成功率伪造。

## 研究背景与动机

随着 AI 生成内容（AIGC）的爆发式增长，不可见图像水印已成为保障内容溯源和责任归属的关键技术。Google、Amazon、OpenAI 等主要 AI 服务商都在积极集成水印系统。然而，**水印伪造攻击**——将可追溯水印伪造到非法内容上——构成对水印系统可信度的严重威胁：
- 攻击者可将伪造带有合法水印的有害图像扩散出去
- 无辜的 AI 服务商会被错误归咎为内容来源
- 导致声誉损害和法律责任

现有伪造攻击主要在黑盒设定下操作（假设可以访问嵌入接口或检测器），但实际中水印嵌入通常集成在生成服务内部，用户无法单独访问。**No-box 设定**（仅能收集到水印图像，不知道水印算法）更贴近现实，但现有方法（Yang et al.）效果极差——它们假设水印信号在所有图像上恒定，通过计算水印图与自然图的平均残差来估计水印，忽略了图像域差异和水印的自适应性。

作者的关键洞察：**扩散模型是天然的分布学习器**，可以隐式捕获训练数据中的水印分布。如果在水印图像上训练无条件扩散模型，模型的去噪过程会自然地将输出引向水印分布。

## 方法详解

### 整体框架

WMCopier 包含三个阶段：
1. **水印估计**：在水印图像上训练无条件扩散模型
2. **水印注入**：通过浅层 DDIM 反演将水印信号注入目标图像
3. **精炼**：迭代优化平衡视觉保真度和水印检测率

### 关键设计

1. **基于扩散模型的水印分布估计**：在辅助数据集 $\mathcal{D}_{\text{aux}} = \{x^w | x^w \sim p_w(x)\}$（5000 张水印图像）上训练无条件扩散模型 $\mathcal{M}_\theta$。核心理论分析：对于水印图 $x^w = x + w$，前向扩散后 $x_t^w = x_t + \sqrt{\alpha_t} w$，噪声预测器的输出：
    $\epsilon_\theta(x_t^w, t) = \hat{\epsilon}(x_t + \sqrt{\alpha_t} w) \approx \hat{\epsilon}(x_t) + \delta_t(w)$
   水印信号 $w$ 引入的预测偏差 $\delta_t(w)$ 在每步去噪中累积，将模型输出分布引向 $p_w(x)$。这意味着模型"学会"了水印的统计特征。

2. **浅层反演注入**：直接使用完整反演（$T_S = T$）会导致严重质量退化，因为 OOD 图像的反演在深层步骤累积大量重建误差。实验发现水印信号主要在浅层步骤（$t \leq 400, T=1000$）中被破坏/恢复。因此只执行浅层反演至 $T_S < T$（默认 $T_S=40, T=100$）：

    - 跳过对水印注入贡献小但严重损害语义的深层扩散步
    - 保留原始图像的视觉保真度
    - 去噪时水印偏差 $\delta_t(w)$ 仍能有效引导生成

3. **迭代精炼**：浅层反演后可能残留轻微伪影。通过梯度上升同时优化水印分布似然和语义保真度：
    $x^{f(i+1)} = x^{f(i)} + \eta \nabla_{x^{f(i)}} \left[\log p_w(x^{f(i)}) - \lambda \|x^{f(i)} - x\|^2 \right]$
   其中 $\log p_w(x^f)$ 用训练好的扩散模型的分数函数近似：
    $\nabla_{x^f} \log p_w(x^f) \approx -\frac{1}{\sqrt{1-\alpha_{t_l}}} \epsilon_\theta(x_{t_l}^f, t_l)$
   $\lambda=100$ 控制语义保持和水印注入的权衡，迭代 $L=100$ 步。

### 损失函数 / 训练策略

- 扩散模型训练：标准去噪目标 $\mathbb{E}[\|\epsilon_\theta(x_t, t) - \epsilon\|_2^2]$
- 辅助数据集：仅需 5000 张水印图像
- DDIM 采样步数 $T=100$，浅层反演至 $T_S=40$
- 精炼参数：$\lambda=100$，$\eta=10^{-4}$，$L=100$，$t_l=1$

## 实验关键数据

### 主实验

**四种开源水印方案攻击对比（平均结果）**

| 攻击方法 | 设定 | PSNR↑ | Forged Bit Acc↑ | FPR@$10^{-6}$↑ |
|---------|------|:---:|:---:|:---:|
| Wang et al. | Black-box | 31.50 | 84.32% | 76.64% |
| Yang et al. | No-box | 30.62 | 54.52% | 0.08% |
| **WMCopier** | **No-box** | **32.94** | **94.58%** | **83.71%** |

**Amazon 商业水印系统攻击**

| 方法 | 数据集 | PSNR↑ | 成功率↑ | 置信度↑ |
|------|--------|:---:|:---:|:---:|
| Yang et al. | DiffusionDB | 23.42 | 29.0% | 2 |
| **WMCopier** | DiffusionDB | **32.57** | **100.0%** | **2.94** |
| Yang et al. | MS-COCO | 24.18 | 32.0% | 2 |
| **WMCopier** | MS-COCO | **32.93** | **100.0%** | **2.97** |

### 消融实验

| 配置 | PSNR | Bit Acc | 说明 |
|------|:---:|:---:|------|
| Full-step inversion ($T_S=T$) | 低 | 高 | 语义内容严重破坏 |
| 浅层反演 ($T_S=40$) | 中 | 中高 | 轻微伪影 |
| 浅层反演 + 精炼 | **高** | **最高** | 伪影消除，质量和效果最优 |
| 精炼迭代 $L$: 0→100 | 逐步↑ | 逐步↑ | $L=100$ 后趋于饱和 |
| 权衡系数 $\lambda$: ↑ | PSNR↑ | Bit Acc略↓ | 过度正则化降低伪造率 |

### 关键发现

- WMCopier 在 no-box 设定下甚至**超越了黑盒攻击**（Wang et al.）的伪造成功率：83.71% vs 76.64%
- 对 Amazon 商业系统近乎 100% 攻击成功率，且置信度接近最高等级（2.94/3）
- 在 HiddeN 方案上伪造 Bit Acc 高达 99.34%，FPR 95.9%
- 伪造水印的鲁棒性略低于真实水印（部分场景退化 10-20%），但通过比特准确率差异无法有效区分真假
- **多消息防御策略有效**：当服务商随机选用 $K=50$ 或 $K=100$ 种水印消息时，WMCopier 的 FPR 降至 0%，即使增加训练数据也无法突破

## 亮点与洞察

- 巧妙地利用了扩散模型的分布学习能力来"复制"水印信号，思路自然且有效
- 浅层反演策略精准地利用了水印信号主要在浅层步骤中被注入/恢复的特性
- 精炼过程中将扩散模型的分数函数作为水印分布的近似，理论推导严谨
- 攻击了真实部署的 Amazon 系统并负责任地报告（附有 Amazon 官方声明），体现了负责任的安全研究态度
- 提出的多消息防御策略简单有效，为工业界提供了实用的对策

## 局限与展望

- 需要收集 5000 张水印图像训练扩散模型，获取成本可能不低
- 假设静态水印方案（服务商不更换算法），动态更新的方案可能更难攻破
- 伪造水印在强扰动下鲁棒性低于真实水印，可被选择性检测
- 多消息防御可完全抵御当前攻击，说明攻击的适用范围有限
- 未考虑语义水印（如 Tree-Ring）的攻击效果

## 相关工作与启发

- 与 T2SMark（本批另一篇）形成"矛与盾"的关系：T2SMark 设计更鲁棒的水印，WMCopier 探索伪造攻击
- 丰富了"扩散模型作为攻击工具"的研究线：此前已有人用扩散模型去除水印，WMCopier 首次用于伪造
- 多消息防御策略的有效性启发水印系统设计应内置抗伪造机制
- 可推广到更多安全领域——用生成模型学习和复制隐蔽信号的一般性框架

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将扩散模型用于 no-box 水印伪造，浅层反演+精炼的设计新颖且有理论支撑
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 4 种开源方案+1 种商业系统、4 个数据集、多组消融和鲁棒性分析，并提出防御
- **写作质量**: ⭐⭐⭐⭐ 论文结构清晰，威胁模型定义严谨，但部分理论推导可更详细
- **价值**: ⭐⭐⭐⭐⭐ 对水印系统安全性提出了重要警告，同时负责任地提出防御方案，兼具学术和工业影响力

<!-- RELATED:START -->

## 相关论文

- [Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models](transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)
- [Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)
- [Detecting Generated Images by Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)

<!-- RELATED:END -->
