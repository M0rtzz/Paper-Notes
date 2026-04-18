---
title: >-
  [论文解读] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation
description: >-
  [CVPR 2026][图像生成][Multi-Human Generation] Ar2Can 提出将多人图像生成分解为空间规划（Architect）和身份保留渲染（Artist）两阶段，通过 GRPO 强化学习配合基于匈牙利匹配的空间锚定人脸奖励函数训练 Artist 模型，在 MultiHuman-Testbench 上实现了 68.2 的身份保留分数和 90.2 的计数准确率，大幅超越所有基线。
tags:
  - CVPR 2026
  - 图像生成
  - Multi-Human Generation
  - Identity Preservation
  - Spatial Planning
  - GRPO
  - 强化学习
---

# Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation

**会议**: CVPR 2026  
**arXiv**: [2511.22690](https://arxiv.org/abs/2511.22690)  
**代码**: [https://qualcomm-ai-research.github.io/ar2can/](https://qualcomm-ai-research.github.io/ar2can/) (有)  
**领域**: 图像生成 / 多人图像生成  
**关键词**: Multi-Human Generation, Identity Preservation, Spatial Planning, GRPO, Reinforcement Learning

## 一句话总结
Ar2Can 提出将多人图像生成分解为空间规划（Architect）和身份保留渲染（Artist）两阶段，通过 GRPO 强化学习配合基于匈牙利匹配的空间锚定人脸奖励函数训练 Artist 模型，在 MultiHuman-Testbench 上实现了 68.2 的身份保留分数和 90.2 的计数准确率，大幅超越所有基线。

## 研究背景与动机

**领域现状**：文生图扩散模型在单人生成方面已非常成熟，但在**多人场景**中系统性失败——身份融合、身份交换、人数错误是普遍问题。

**现有方法的分类与痛点**：
   - **区域条件方法**（GLIGEN, ReCo）需要用户手动提供空间标注，可用性差
   - **身份保留方法**（IP-Adapter, InstantID, PuLID）适用于单人，多人时身份冲突
   - **多 ID 方法**（OmniGen, DreamO, XVerse）在最新 benchmark 上仍表现不佳

**核心矛盾**：现有方法将**空间布局推理**和**身份渲染**融合在单一生成过程中。当模型需要同时决定"人在哪里"和"人长什么样"时，空间结构与外观纠缠，导致身份融合。

**切入角度**：**解耦空间规划与身份渲染**——先确定每个人出现在哪，再专注于逼真渲染。这种分治策略能从根本上避免身份融合。

**核心 idea**：Architect 生成结构化空间布局（bounding box / 姿态），Artist 在此布局指导下通过 GRPO + 匈牙利匹配人脸奖励来保持多身份一致性。

## 方法详解

### 整体框架
两阶段：Architect → Canvas → Artist
- 输入：文本提示 $p$ + $N$ 张参考身份图 $\{I_{ref,1}, ..., I_{ref,N}\}$
- Architect：预测空间布局 $\mathcal{L} = \{b_1, ..., b_N\}$（每人一个 bounding box）
- Canvas 构建：将参考人脸粘贴到对应位置
- Artist：以 canvas + 参考图 + 文本为条件，生成最终图像

### 关键设计

1. **Architect-A（LLM 方案）**：

    - 基于 Qwen-2.5-0.5B-Instruct 微调
    - 扩展词表加入 `<SoL>`, `<EoL>`, `<C>` 等布局结构令牌
    - 同时预测令牌序列和连续坐标（双头设计：$f_{token}$ + $f_{value}$）
    - 损失：$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{coord}[\mathcal{L}_{gIoU} + \|b_{pred} - b_{gt}\|_1]$
    - **优势**：强语言理解能力，人数准确率最高（90.2%）

2. **Architect-B（T2I 方案）**：

    - 基于 Flux-Schnell 微调（仅需 4 步去噪）
    - 用 GRPO 强化学习训练，奖励函数：$r = \alpha \cdot r_{count} + \beta \cdot r_{hps}$
    - 能同时输出人脸 bounding box 和人体姿态
    - **优势**：空间先验更强，动作分数更高

3. **Artist 的 GRPO 训练与组合奖励**：

    - 基于 Flux-Kontext 训练，使用四个组合奖励：
    $r_{Artist} = \alpha \cdot r_{count} + \beta \cdot r_{hps} + \zeta \cdot r_{face} + \eta \cdot r_{pose}$
    - **空间锚定人脸匹配奖励 $r_{face}$（核心创新）**：
        - 第一步：用**匈牙利算法**在 Architect 预测的中心点和 RetinaFace 检测到的人脸中心点之间做最优匹配
        - 第二步：对匹配的人脸对计算 ArcFace 余弦相似度
        - **设计动机**：朴素的位置匹配（直接在预测位置裁剪人脸）会导致 copy-paste 伪影和奖励黑客攻击——模型直接在精确位置粘贴参考人脸。匈牙利中心点匹配放松了空间约束，允许自然变化

4. **Token Sharing & Dropping**：

    - 丢弃 canvas 中非信息区域的 token，平均减少 2× token 数量
    - 当人脸 bounding box 重叠时，共享相同的 RoPE 位置编码
    - **设计动机**：共享位置编码迫使模型学习遮挡处理（深度分层、空间重排），而非简单粘贴

5. **课程学习**：前 $\tau=100$ 个 epoch 只用 2-3 人场景，之后均匀采样 2-7 人场景。避免早期训练崩溃。

### 训练数据
- 核心挑战：缺乏大规模多人训练数据
- 使用 DisCo 生成合成多人场景，与真实参考人脸配对构建混合训练样本
- 完全使用合成多人数据，无需真实多人照片

## 实验关键数据

### 主实验（MultiHuman-TestBench）
| 方法 | Count↑ | Multi-ID↑ | HPS↑ | Action-S↑ | Unified↑ |
|------|--------|-----------|------|-----------|----------|
| GPT-Image-1 | 87.9 | 28.8 | 30.3 | 97.0 | 55.8 |
| DreamO | 61.2 | 34.7 | 28.5 | 86.2 | 59.7 |
| MH-OmniGen | 60.3 | 54.5 | 26.3 | 91.6 | 61.6 |
| XVerse | 81.7 | 30.6 | 25.5 | 66.2 | 52.7 |
| **Ar2Can (Arch-B)** | 86.9 | **68.2** | **30.8** | 86.2 | **72.4** |
| **Ar2Can (Arch-A)** | **90.2** | 67.6 | 30.2 | 86.3 | 72.2 |

### 消融实验
| 配置 | Count↑ | Multi-ID↑ | HPS↑ | 说明 |
|------|--------|-----------|------|------|
| Baseline (Kontext) | 80.7 | 14.5 | 29.2 | 原始模型，多人严重失败 |
| + Simple Matching | 75.6 | 55.2 | 27.6 | 朴素匹配→copy-paste伪影 |
| + Hungarian Centroid | 80.1 | 60.3 | 30.9 | 匈牙利匹配恢复质量 |
| + Curriculum (Full) | **86.9** | **68.2** | **30.8** | 课程学习进一步提升 |

### 关键发现
- Multi-ID 从 Kontext 的 14.5 提升到 68.2（+53.7），说明 Kontext 在多人场景下几乎完全失败
- 匈牙利匹配相比朴素匹配不仅提升身份保留（+5.1），还恢复了图像质量（HPS 27.6→30.9）
- 在 88% 的评测提示中被人类评估者优选（vs DreamO 4%, XVerse 8%）
- 仅用合成数据训练就超越了使用大量真实数据的商业模型

## 亮点与洞察
- **空间规划与渲染解耦**是一个非常清晰的思路，有效避免了多人生成中的身份纠缠问题
- **匈牙利匹配奖励**巧妙地平衡了空间精度和生成自然度——不硬性要求精确位置，而是"附近即可"
- **模块化设计**的灵活性：Architect 可替换不同方案而不需重新训练 Artist
- 在主要使用合成数据的情况下取得 SOTA，说明 RL 微调可以有效补偿数据质量

## 局限与展望
- 动作生成方面仍不如 GPT-Image-1（Action-S 86.2 vs 97.0），序复杂动作理解有待加强
- 7 人以上的极端多人场景未验证
- 推理延迟较高（两阶段架构带来额外开销），token sharing 虽缓解但仍有空间
- Architect-A 和 Architect-B 各有优劣，缺乏统一方案

## 相关工作与启发
- DisCo（Flow-GRPO 用于多人生成）是直接前驱，Ar2Can 在其基础上加入了空间锚定
- Canvas-based 方法（如 Kontext）在单人时效果好但多人失败，说明需要显式空间引导
- GRPO + 组合奖励的范式可推广到其他需要多目标平衡的图像生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 空间解耦+匈牙利匹配奖励的组合新颖且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 两个 benchmark + 人类评估 + 详细消融 + 延迟分析
- 写作质量: ⭐⭐⭐⭐ 框架清晰，但部分细节需查附录
- 价值: ⭐⭐⭐⭐⭐ 多人身份保留生成是刚需，实际改进巨大（+13.7 Multi-ID over SOTA）
