---
title: >-
  ICML2026 图像恢复方向2篇论文解读
description: >-
  2篇ICML2026的图像恢复方向论文解读，涵盖超分辨率、扩散模型、图像恢复等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "超分辨率"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (39)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (14)](../../ICLR2026/image_restoration/index.md) · [🤖 AAAI2026 (13)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (29)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

**[Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)**

:   H-VAR 把"残差量化做多尺度生成"的 VAR 范式重新切片成层次化的图像 tokenization (HIT)，让一个 310M 的小模型只跑一次前向就能输出 128 / 256 / 512 三个有意义的中间分辨率，再配一个不需要外部奖励模型的 DPO 正则项推动输出偏向 HR，在标准 ISR 数据上对打 1B 参数的 VARSR。

**[Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)**

:   SubDAPS / SubDAPS++ 把 DPS、DAPS 这类 pixel-space 扩散复原方法搬进"动态分辨率扩散模型"框架——早期在 $64^2 / 128^2$ 子空间采样、后期才回到 $256^2$ 全分辨率，并用共轭梯度替掉 Langevin、用阈值切换 stochastic / deterministic 采样、再附一个无需额外网络评估的 corrector 步，在 4 类线性 + 2 类非线性复原任务上多数指标超越 pixel 与 latent 扩散方法且推理更快。
