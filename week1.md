# 周报1 |（7.23 - 8.1）

### Transformer 架构
- **Self-Attention**：通过计算 Query-Key 的 dot product，获取 token 间关系权重。
- **位置编码（Positional Encoding）**：使用正弦/余弦函数嵌入位置信息，补足无序特性。
- 绘制了 attention 计算公式图解。

### BERT vs GPT 架构对比
| 模型 | 架构类型       | 预训练任务            | 适用方向     |
|------|----------------|------------------------|--------------|
| BERT | Encoder-only   | MLM + NSP              | 表征学习     |
| GPT  | Decoder-only   | 自回归语言建模        | 文本生成     |

- **BERT 阅读重点**：Masked Language Modeling（MLM）和 Next Sentence Prediction（NSP）。

---

## Scikit-learn尝试
- **数据集**：Iris
- **模型**：
  - LogisticRegression
  - DecisionTreeClassifier
- **超参数**：
  - 决策树：`max_depth=3`
  - 回归器：默认
- **准确率结果**：逻辑回归（97.3%），决策树（96.0%）

## Word2Vec多语言对比分析
- **数据集**：中日英三个语言版本的《我是猫》文本
- **模型**：
  - gensim.models.Word2Vec
- **参数设置**：
  - vector_size=50：词向量维度为 50
  - window=3：上下文窗口大小为 3
  - min_count=1：所有词都参与训练（不舍弃低频词）
  - sample=1e-5：高频词的下采样阈值
---

## 水印方法学习（具体内容见笔记）
### A Watermark for LLMs（2023）
- **核心思想**：在生成阶段调整 logits，使 greenlist tokens 概率增强，嵌入统计性偏差。
- **关键方法**：
  - 设定 secret key 与 hash 函数 h()。
  - 每步选出 greenlist，强行提升其概率。
- **检测**：z-test 检测 green token 占比是否显著偏高。

### Scalable Watermarking for Identifying LLM Outputs（SynthID-Text，2024）
- **新方法**：Tournament Sampling（多层胜出淘汰制）。
- **水印嵌入流程**：
  1. 从 pLM 采样多个 token。
  2. 用随机函数 g₁...gₘ 对 token 赋值。
  3. 分层次优胜者晋级，最终决出输出 token。
- **检测方法**：
  \[
  \text{Score}(x) = \frac{1}{mT} \sum_{t=1}^T \sum_{\ell=1}^m g_\ell(x_t, r_t)
  \]
  - 利用得分偏高进行判别。


