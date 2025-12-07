# Disaster Tweets Classification

基于 BERTweet/BERT 的灾难推文分类模型，用于 [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) 比赛。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载数据

```bash
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data/
```

### 训练 & 预测

```bash
# 完整流程（训练 + 预测）
python run.py full --model bertweet --epochs 5

# 仅训练
python run.py train --model bertweet --epochs 5

# 仅预测
python run.py predict --model_dir outputs/bertweet_xxx/best_model
```

## 可用模型

| 模型 | 说明 |
|------|------|
| `bertweet` | BERTweet base（推荐，针对Twitter优化） |
| `bertweet-large` | BERTweet large |
| `bert` | BERT base uncased |
| `bert-large` | BERT large uncased |
| `roberta` | RoBERTa base |

## 项目结构

```
├── run.py              # 主入口
├── src/
│   ├── train.py        # 训练脚本
│   ├── inference.py    # 推理脚本
│   └── dataset.py      # 数据处理
├── data/               # 数据目录
├── outputs/            # 模型输出
└── submissions/        # 提交文件
```

## 实验结果

详见 [EXPERIMENTS.md](EXPERIMENTS.md)

| 模型 | Val F1 | Val Acc | 备注 |
|------|--------|---------|------|
| BERTweet Base | **0.8337** | 0.8599 | 最佳 Epoch 2 |

