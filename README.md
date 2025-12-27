# Disaster Tweets Classification

灾难推文分类项目，用于 [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) 比赛。

本项目探索了三种不同的深度学习方法来解决灾难推文分类任务。

## 三种实现方法对比

本项目包含三个不同日期的实现，每个代表不同的深度学习方法：

| 日期 | 文件夹 | 方法 | 核心技术 | 特点 |
|------|--------|------|----------|------|
| **12.7** | `12.7_bert/` | BERT/BERTweet微调 | Transformer全参数微调 | 传统微调方法，需要GPU训练 |
| **12.15** | `12.15_qwen_lora/` | Qwen2.5 + LoRA | 参数高效微调(PEFT) | 低资源微调，仅训练少量参数 |
| **12.25** | `12.25_llm_api/` | LLM API + Few-Shot Learning | 大模型API调用 + Prompt工程 | 无需训练，通过精心设计的提示词实现 |

### 方法详解

#### 12.7 - BERT微调方法
- **技术**: 使用 BERTweet/BERT 进行全参数微调
- **优势**: 针对任务定制化训练，效果稳定
- **劣势**: 需要GPU资源，训练时间较长
- **适用场景**: 有充足计算资源，追求最优性能

#### 12.15 - LoRA参数高效微调
- **技术**: 使用 Qwen2.5-1.5B + LoRA（低秩适应）
- **优势**: 仅训练<1%参数，显存占用低，训练速度快
- **劣势**: 需要支持 LoRA 的框架
- **适用场景**: GPU显存有限，需要快速迭代

#### 12.25 - LLM API调用方法
- **技术**: 调用外部大模型API（DeepSeek-V3），使用 Few-Shot Learning
- **优势**: 无需训练，无需GPU，部署简单，灵活调整
- **劣势**: 依赖网络和API，推理成本按token计费
- **适用场景**: 快速原型验证，无GPU环境，小规模推理

## 实验结果

详见 [EXPERIMENTS.md](EXPERIMENTS.md)

| 模型 | Val F1 | Val Acc | 备注 |
|------|--------|---------|------|
| BERTweet Base (12.7) | **0.8337** | 0.8599 | 最佳 Epoch 2 |
| Qwen2.5-1.5B LoRA (12.15) | TBD | TBD | LoRA微调 |
| DeepSeek-V3 Few-Shot (12.25) | TBD | TBD | 8-shot learning |


## 快速开始

### 方法一：BERT微调 (12.7)

```bash
cd 12.7_bert
pip install -r requirements.txt

# 训练 + 预测
python run.py full --model bertweet --epochs 5

# 仅训练
python run.py train --model bertweet --epochs 5

# 仅预测
python run.py predict --model_dir outputs/bertweet_xxx/best_model
```

#### 可用模型
- `bertweet`: BERTweet base（推荐，针对Twitter优化）
- `bert`: BERT base uncased
- `roberta`: RoBERTa base

### 方法二：Qwen LoRA微调 (12.15)

```bash
cd 12.15_qwen_lora
pip install -r requirements.txt

# 运行 LoRA 训练和推理
python run_lora_inference.py
```

### 方法三：LLM API调用 (12.25)

```bash
cd 12.25_llm_api
pip install -r requirements.txt

# 基础推理（8-shot）
python run_api_inference.py

# 带验证集评估
python run_api_inference.py --evaluate --val_size 100

# 自定义配置
python run_api_inference.py --n_examples 16 --temperature 0.2
```

### 下载数据（所有方法通用）

```bash
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data/
```

## 项目结构

```
├── 12.7_bert/                    # BERT微调方法（12月7日）
│   ├── run.py                    # 训练和推理入口
│   ├── src/
│   │   ├── train.py              # 训练脚本
│   │   ├── inference.py          # 推理脚本
│   │   └── dataset.py            # 数据处理
│   └── requirements.txt
│
├── 12.15_qwen_lora/              # Qwen LoRA微调方法（12月15日）
│   ├── run_lora_inference.py     # LoRA训练和推理
│   ├── src/
│   │   ├── train_lora.py         # LoRA训练脚本
│   │   └── correct_predictions.py
│   └── requirements.txt
│
├── 12.25_llm_api/                # LLM API调用方法（12月25日）
│   ├── run_api_inference.py      # API推理入口
│   ├── src/
│   │   └── llm_classifier.py     # Few-shot分类器
│   ├── requirements.txt
│   └── README.md                 # 详细说明
│
├── data/                         # 共享数据目录
├── outputs/                      # 模型输出
└── submissions/                  # 提交文件
```

## 技术栈对比

| 技术栈 | 12.7 BERT | 12.15 Qwen LoRA | 12.25 LLM API |
|--------|-----------|-----------------|---------------|
| **框架** | Transformers + PyTorch | Transformers + PEFT | Requests |
| **模型大小** | 110M-340M参数 | 1.5B参数（训练<1%） | ~671B参数（API） |
| **显存需求** | 4-8GB | 8-12GB | 0GB（云端） |
| **训练时间** | 数小时 | 1-2小时 | 无需训练 |
| **推理速度** | 快（本地） | 快（本地） | 慢（网络延迟） |
| **成本** | GPU租赁 | GPU租赁 | API调用费用 |

## 方法选择建议

- **追求最高准确率**: 选择 12.7 BERT微调，有充足GPU资源
- **资源受限但有GPU**: 选择 12.15 Qwen LoRA，快速迭代
- **快速原型/无GPU**: 选择 12.25 LLM API，立即部署
- **生产环境**: 结合使用，12.25快速验证 → 12.15/12.7优化部署



