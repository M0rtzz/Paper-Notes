# DiVoT: Diffusion Powers Video Tokenizer for Comprehension and Generation

**会议**: CVPR 2025  
**arXiv**: [2412.04432](https://arxiv.org/abs/2412.04432)  
**代码**: https://github.com/TencentARC/Divot  
**领域**: 扩散模型 / 视频理解  
**关键词**: 视频分词器, 扩散去噪, 统一理解生成, 高斯混合模型, LLM视频

## 一句话总结
提出 DiVoT，用扩散去噪过程作为视频分词器的自监督训练信号——若分词器的连续表示能引导 U-Net 成功去噪视频则表示有效，统一视频理解和生成，LLM 通过高斯混合模型（而非 MSE 回归）预测连续视频特征。

## 研究背景与动机

**领域现状**：统一视频理解和生成需要有效的视频表示。离散 token（VQVAE）丢失信息，连续表示（CLIP 特征等）缺乏时空细节。现有统一模型（Video-LaVIT, VILA-U）在理解或生成上难以兼顾。

**现有痛点**：(1) 离散 token 的信息瓶颈限制了生成质量；(2) 连续特征（如 CLIP）缺乏密集时空信息，不适合视频生成；(3) LLM 预测连续特征时 MSE 回归产生过度平均的模糊特征。

**核心矛盾**：视频表示需要同时包含语义（理解）和密集时空（生成）信息，且 LLM 需要能建模其分布而非仅预测均值。

**切入角度**：扩散去噪本身就是强大的自监督信号——如果连续表示能引导去噪成功，说明它捕获了足够的时空信息。用 GMM 替代 MSE 让 LLM 建模多模态分布。

**核心 idea**：扩散去噪作为自监督信号训练视频分词器 + GMM 分布建模替代 MSE 回归 + U-Net 兼做反分词器。

## 方法详解

### 关键设计

1. **扩散驱动分词器**：ViT + 时空 Transformer + Perceiver Resampler 编码稀疏帧为连续表示，条件化视频 U-Net 去噪密集帧。U-Net 同时作为反分词器（解码器）

2. **GMM 特征建模**：LLM 输出 GMM 参数（均值、方差、混合权重）而非单一特征向量，采样时从 GMM 抽取更多样化的特征

3. **统一架构**：Mistral-7B 作为骨干，理解时直接输出文本，生成时输出 GMM 参数→采样特征→U-Net 解码

## 实验关键数据

| 基准 | DiVoT | Video-LaVIT | VILA-U |
|------|-------|------------|--------|
| EgoSchema | **46.5** | 34.4 | 37.4 |
| PerceptionTest | **58.3** | 44.6 | 54.0 |
| MVBench | **52.1** | — | 44.0 |

仅用 4.8M 视频文本对训练（竞品用 100M+），理解性能全面领先统一模型。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 扩散去噪作为分词信号 + GMM 分布建模双重创新
- 实验充分度: ⭐⭐⭐⭐ 理解+生成多基准
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 统一视频理解+生成的新范式

<!-- RELATED:START -->

## 相关论文

- [GraphGPT-o: Synergistic Multimodal Comprehension and Generation on Graphs](graphgpt-o_synergistic_multimodal_comprehension_and_generation_on_graphs.md)
- [TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [Holistic Tokenizer for Autoregressive Image Generation](../../ICCV2025/image_generation/holistic_tokenizer_for_autoregressive_image_generation.md)
- [EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)
- [Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)

<!-- RELATED:END -->
