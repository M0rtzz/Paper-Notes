---
title: >-
  [论文解读] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing
description: >-
  [CVPR 2026][图像生成][多人运动编辑] 提出 InterEdit，首个文本引导的多人3D运动编辑框架，通过语义感知 Plan Token 对齐和交互感知频域 Token 对齐两个机制，在条件扩散模型中实现对双人交互动作的精准编辑，同时保持源运动的一致性和交互协调性。
tags:
  - CVPR 2026
  - 图像生成
  - 多人运动编辑
  - 文本引导
  - 交互感知
  - 频域对齐
  - 条件扩散模型
---

# InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing

**会议**: CVPR 2026  
**arXiv**: [2603.13082](https://arxiv.org/abs/2603.13082)  
**代码**: [GitHub](https://github.com/YNG916/InterEdit)  
**领域**: 3D 运动生成 / 扩散模型  
**关键词**: 多人运动编辑, 文本引导, 交互感知, 频域对齐, 条件扩散模型

## 一句话总结

提出 InterEdit，首个文本引导的多人3D运动编辑框架，通过语义感知 Plan Token 对齐和交互感知频域 Token 对齐两个机制，在条件扩散模型中实现对双人交互动作的精准编辑，同时保持源运动的一致性和交互协调性。

## 研究背景与动机

文本引导的单人运动编辑已取得显著进展，但扩展到多人场景面临独特挑战：

**配对数据稀缺**：缺乏（源运动, 目标运动, 编辑指令）三元组形式的多人运动编辑数据

**交互语义复杂**：运动含义不仅来自个体动作，还来自时空耦合——同步、相位对齐、角色切换、接触时机等

**编辑约束更严**：需要"改请求的部分，保其余部分"，而在交互场景中，微小的时间偏移即可改变语义

**核心 gap**：现有单人编辑方法（MotionFix、MotionLab）忽略交互耦合，直接拼接双人特征会破坏协调性；多人生成方法（InterGen、TIMotion）缺乏"什么该改、什么该保"的机制。目前没有专门的多人运动编辑基准。

## 方法详解

### 整体框架

InterEdit 是一个条件扩散框架，以 Start_X 参数化直接预测干净运动 $\hat{\mathbf{x}}_0 = \mathcal{D}_\theta(\mathbf{x}_t, t; \mathbf{c}_{\text{text}}, \mathbf{c}_{\text{src}})$。骨干采用 Transformer-based denoiser，条件通过 AdaLN 注入：

$$\mathbf{e}_t = \mathrm{EmbedTime}(t) + W_{\text{text}}\mathbf{c}_{\text{text}} + W_{\text{src}}\mathbf{c}_{\text{src}}$$

核心创新在两个辅助对齐机制：Semantic-Aware Plan Token Alignment 和 Interaction-Aware Frequency Token Alignment。

### 关键设计

1. **Symmetric Interleaved Token Aggregation（基础架构）**

   构建因果交错序列建模双人时序影响和角色切换。对 A/B 两人的运动 token $\mathbf{x}_c^A, \mathbf{x}_c^B \in \mathbb{R}^{L \times C}$，构建交错序列 $\mathbf{x}_{\mathrm{cii}}$ 和角色互换对称序列 $\mathbf{x}_{\mathrm{sym}}$：

   $$\mathbf{x}_{\mathrm{cii}}(2\ell-1) = \mathbf{x}_c^A(\ell), \quad \mathbf{x}_{\mathrm{cii}}(2\ell) = \mathbf{x}_c^B(\ell)$$

   拼接后经 Transformer 处理，再反交错+角色视角融合得到全局特征，辅以 LPA（Localized Pattern Amplification）分支提取短程时间模式。

2. **Semantic-Aware Plan Token Alignment（语义引导）**

   附加 $N_M=16$ 个可学习 Plan Token $\mathbf{P} \in \mathbb{R}^{N_M \times 2C}$ 到 denoiser 序列。在 Transformer block $L_p$ 处投射到语义空间并与冻结运动教师编码器提取的目标运动嵌入 $\mathbf{z}_{\text{tgt}} = f_T(\mathbf{x}_0)$ 对齐：

   $$\mathcal{L}_{\text{plan}} = \frac{1}{N_M}\sum_{k=1}^{N_M}\left[-\log\frac{\exp((\tilde{\mathbf{z}}^{(k)})^\top \tilde{\mathbf{z}}_{\text{tgt}} / \tau)}{\sum_n \exp((\tilde{\mathbf{z}}^{(k)})^\top \tilde{\mathbf{z}}_{\text{tgt}}^{(n)} / \tau)}\right]$$

   通过 InfoNCE 损失对齐，Plan Token 在自注意力中为运动 token 提供高层编辑语义引导。

3. **Interaction-Aware Frequency Token Alignment（交互动力学）**

   构建交互信号：均值 $\mathbf{z}_S = (\mathbf{x}^A + \mathbf{x}^B)/2$（同步分量）和差值 $\mathbf{z}_D = \mathbf{x}^A - \mathbf{x}^B$（对抗分量），对其进行 DCT 变换并按低/中/高三频段池化得到6个频带能量描述符：

   $$\mathbf{E}(\mathbf{C};b) = \sqrt{\frac{1}{|b|}\sum_{k \in b} \mathbf{C}[k]^2 + \epsilon}$$

   将频带能量投射为6个 Frequency Token 注入序列，在 block $L_f$ 处解码并以加权回归损失对齐目标运动的频带能量：$\mathcal{L}_{\text{freq}} = \frac{1}{N_f}\sum_i w_i \|\hat{\mathbf{g}}_i - \mathbf{g}_i(\mathbf{x}_0)\|_2^2$。训练时高频项降权0.25，频率 token 以概率 $p_f=0.04$ 随机丢弃防过拟合。

### 损失函数 / 训练策略

总目标函数：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{motion}} + \lambda_p \mathcal{L}_{\text{plan}} + \lambda_f \mathcal{L}_{\text{freq}}$

其中运动损失包含：扩散重建 $\mathcal{L}_{\text{diff}}$、速度 $\mathcal{L}_{\text{vel}}$、脚接触 $\mathcal{L}_{\text{foot}}$、骨长 $\mathcal{L}_{\text{BL}}$ + 交互损失（距离图 $\mathcal{L}_{\text{DM}}$、相对朝向 $\mathcal{L}_{\text{RO}}$）。$\lambda_p = 0.03$，$\lambda_f = 0.01$。推理时采用 Synchronized CFG（$\gamma=3.5$），同步丢弃文本和源运动条件。DDIM 50步采样。358.8M 参数（85.0M 可训练），8卡 RTX Pro 6000 训练1500 epochs。

## 实验关键数据

### 主实验

InterEdit3D 测试集评测（5161个三元组，80/10/10划分）：

| 方法 | FID↓ | g2s R@1↑ | g2s R@3↑ | g2t R@1↑ | g2t R@3↑ |
|------|------|---------|---------|---------|---------|
| MotionFix | 2.547 | 2.51 | 6.76 | 3.86 | 7.73 |
| MotionLab | 0.550 | 7.90 | 16.43 | 13.26 | 20.69 |
| InterGen | 0.624 | 9.52 | 18.91 | 18.93 | 31.64 |
| TIMotion | 0.445 | 12.54 | 22.33 | 24.97 | 40.68 |
| **InterEdit** | **0.371** | **17.08** | **29.32** | **30.82** | **47.65** |

相比最强基线 TIMotion：g2t R@1/2/3 分别提升 +5.85/+7.07/+6.97，FID 降低 16.7%。

### 消融实验

| 配置 | FID↓ | g2t R@1↑ | g2t R@3↑ |
|------|------|---------|---------|
| w/o plan + freq | 0.445 | 24.97 | 40.68 |
| only plan token | 0.367 | 28.72 | 43.50 |
| only freq token | 0.380 | 28.75 | 44.05 |
| **plan + freq (full)** | **0.371** | **30.82** | **47.65** |

频率 token dropout 率消融（$p_f$=0.04 最优，平衡过拟合与信号强度）。

### 关键发现

- 多人生成基线（InterGen/TIMotion）显著优于单人编辑基线，证实交互建模的必要性
- Plan Token 和 Frequency Token 功能互补：前者引导"改什么"，后者稳定"怎么改"
- 联合使用两者的提升大于各自单独使用之和（g2t R@3: 40.68→43.50/44.05→47.65）

## 亮点与洞察

- **首创多人运动编辑任务和基准**：填补了该领域空白，InterEdit3D 含 5161 个高质量三元组
- **频域交互建模**：DCT 分解+频带能量描述符优雅地捕捉了交互的节奏和同步特性
- **Plan Token 的 InfoNCE 对齐**：无需显式标注"哪些关节该改"，通过对比学习自动获取编辑意图

## 局限与展望

- 仅支持双人交互，扩展到三人及以上场景需重新设计交错策略
- 依赖 InterHuman 数据集的动作类型范围（日常活动+武术/舞蹈），更多场景需扩展数据
- 运动表示基于关节坐标，缺乏外观/形态信息

## 相关工作与启发

- **MotionFix**：单人运动编辑开创者，本文将其扩展到多人
- **TIMotion**：最强多人生成基线，InteEdit 复用了其对称交错 token 和 LPA 设计
- **TMR**：对比训练的运动编码器，作为 Plan Token 对齐的冻结教师

## 评分

- **新颖性**: ★★★★☆ — 任务定义+频域交互对齐是新贡献
- **技术深度**: ★★★★☆ — Plan/Frequency Token 双轴设计完整，损失函数丰富
- **实验充分度**: ★★★★☆ — 定量+定性+消融全面，但基线都是适配后的非原生方法
- **实用性**: ★★★☆☆ — 研究驱动型工作，数据集和代码即将开源

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)
- [EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)

<!-- RELATED:END -->
