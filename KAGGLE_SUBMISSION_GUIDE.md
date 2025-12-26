# Kaggle 提交指南

本指南介绍如何生成提交文件并上传到 Kaggle 竞赛。

## 步骤一：生成提交文件

### 方法 1：使用标准分类模型（BERT/BERTweet）

如果你的模型是使用 `src/train.py` 训练的 BERT/BERTweet 等分类模型：

```bash
# 使用 run.py（推荐）
python run.py predict --model_dir outputs/bertweet_xxx/best_model

# 或直接使用 inference.py
python src/inference.py \
    --model_dir outputs/bertweet_xxx/best_model \
    --data_dir ./data \
    --output_dir ./submissions \
    --output_name submission.csv \
    --batch_size 64
```

### 方法 2：使用 LoRA 训练的生成式模型

如果你的模型是使用 `src/train_lora.py` 训练的 LoRA 模型（Qwen/Phi/Llama等），需要创建专门的推理脚本。

**注意**：当前项目中的 `inference.py` 不支持 LoRA 模型。你需要：

1. 加载基础模型和 LoRA 适配器
2. 使用生成式推理（而非分类）
3. 解析生成结果（yes/no 或 disaster/not disaster）

## 步骤二：验证提交文件格式

提交文件必须是 CSV 格式，包含两列：
- `id`: 测试样本的 ID
- `target`: 预测标签（0 或 1）

示例：
```csv
id,target
0,0
2,1
3,0
9,1
...
```

验证文件格式：
```bash
# 查看提交文件前几行
head -10 submissions/submission.csv

# 检查文件行数（应该等于测试集大小 + 1 行表头）
wc -l submissions/submission.csv

# 验证格式（Python）
python -c "import pandas as pd; df = pd.read_csv('submissions/submission.csv'); print(df.head()); print(f'Shape: {df.shape}'); print(f'Target values: {df.target.unique()}')"
```

## 步骤三：提交到 Kaggle

### 方法 1：通过网页提交（最简单）

1. **访问竞赛页面**
   - 打开 https://www.kaggle.com/competitions/nlp-getting-started

2. **进入提交页面**
   - 点击页面上的 **"Submit Predictions"** 或 **"Late Submission"** 按钮

3. **上传文件**
   - 点击 **"Upload Submission File"** 或 **"Browse"** 按钮
   - 选择你的 `submission.csv` 文件
   - 等待上传完成

4. **查看结果**
   - 上传后，Kaggle 会自动验证文件格式
   - 如果格式正确，会显示你的分数（F1 Score）
   - 可以在 **"My Submissions"** 页面查看历史提交

### 方法 2：使用 Kaggle API 提交

1. **安装 Kaggle API**（如果还没安装）
   ```bash
   pip install kaggle
   ```

2. **配置 API 凭证**（如果还没配置）
   ```bash
   # 将 kaggle.json 放在 ~/.kaggle/ 目录
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **提交文件**
   ```bash
   kaggle competitions submit -c nlp-getting-started -f submissions/submission.csv -m "My submission message"
   ```

4. **查看提交状态**
   ```bash
   kaggle competitions submissions -c nlp-getting-started
   ```

## 步骤四：查看提交结果

### 在网页上查看

1. 访问竞赛页面
2. 点击 **"My Submissions"** 或 **"Submissions"** 标签
3. 查看你的提交历史、分数和排名

### 使用 API 查看

```bash
# 列出所有提交
kaggle competitions submissions -c nlp-getting-started

# 查看最新提交的详细信息
kaggle competitions submissions -c nlp-getting-started --max 1
```

## 提交文件命名建议

建议使用有意义的文件名，方便追踪：

```bash
# 示例命名
submission_bertweet_epoch5.csv
submission_qwen_lora_stage2.csv
submission_ensemble_v1.csv
```

## 常见问题

### 1. 提交文件格式错误

**错误信息**：`Submission file must have exactly 2 columns: id, target`

**解决方法**：
- 确保 CSV 文件只有两列：`id` 和 `target`
- 确保没有额外的空格或特殊字符
- 确保使用逗号分隔，不是分号或制表符

```python
# 验证并修复格式
import pandas as pd

df = pd.read_csv('submissions/submission.csv')
print(df.columns)  # 应该显示 ['id', 'target']
print(df.dtypes)   # id 应该是 int64, target 应该是 int64
print(df['target'].unique())  # 应该只有 [0, 1]

# 如果 target 是 float，转换为 int
if df['target'].dtype == float:
    df['target'] = df['target'].astype(int)

# 保存修复后的文件
df.to_csv('submissions/submission_fixed.csv', index=False)
```

### 2. 提交文件行数不匹配

**错误信息**：`Submission file must have exactly X rows`

**解决方法**：
- 确保提交文件的行数 = 测试集大小 + 1（表头）
- 检查是否有重复或缺失的 ID

```python
# 检查行数
import pandas as pd

test_df = pd.read_csv('data/test.csv')
submission_df = pd.read_csv('submissions/submission.csv')

print(f"Test set size: {len(test_df)}")
print(f"Submission size: {len(submission_df)}")
print(f"Expected submission size: {len(test_df) + 1}")  # +1 for header

# 检查 ID 是否匹配
missing_ids = set(test_df['id']) - set(submission_df['id'])
if missing_ids:
    print(f"Missing IDs: {missing_ids}")
```

### 3. 目标值超出范围

**错误信息**：`Target values must be 0 or 1`

**解决方法**：
- 确保所有 `target` 值都是 0 或 1
- 检查是否有 NaN 值

```python
# 修复目标值
import pandas as pd
import numpy as np

df = pd.read_csv('submissions/submission.csv')

# 检查并修复
print(f"Unique values: {df['target'].unique()}")
print(f"NaN count: {df['target'].isna().sum()}")

# 如果有 NaN，填充为 0
df['target'] = df['target'].fillna(0)

# 如果有超出范围的值，截断
df['target'] = df['target'].clip(0, 1).astype(int)

# 保存
df.to_csv('submissions/submission_fixed.csv', index=False)
```

### 4. 提交次数限制

- 大多数 Kaggle 竞赛每天有提交次数限制（通常 5-10 次）
- 如果达到限制，需要等待第二天再提交
- 可以在竞赛页面的 "Rules" 部分查看具体限制

### 5. 竞赛已结束

- 如果竞赛已结束，可能只能进行 "Late Submission"
- Late Submission 通常不计入排行榜，但可以查看分数

## 最佳实践

1. **本地验证**
   - 在提交前，先在本地验证文件格式
   - 检查预测分布是否合理（不应该全是 0 或全是 1）

2. **版本控制**
   - 为每次提交保存文件副本
   - 记录使用的模型和参数

3. **提交信息**
   - 使用有意义的提交消息，方便追踪
   - 例如：`BERTweet base, epoch 5, F1=0.8337`

4. **多次提交**
   - 可以尝试不同的模型或参数
   - 记录每次提交的分数，找出最佳模型

## 示例：完整提交流程

```bash
# 1. 生成预测
python run.py predict --model_dir outputs/bertweet_20231207_120000/best_model

# 2. 验证文件
python -c "
import pandas as pd
df = pd.read_csv('submissions/submission.csv')
print(f'Shape: {df.shape}')
print(f'Target distribution: {df.target.value_counts()}')
print(f'Target unique: {df.target.unique()}')
"

# 3. 提交到 Kaggle
kaggle competitions submit -c nlp-getting-started \
    -f submissions/submission.csv \
    -m "BERTweet base, epoch 5"

# 4. 查看提交状态
kaggle competitions submissions -c nlp-getting-started --max 1
```

## 参考链接

- [竞赛页面](https://www.kaggle.com/competitions/nlp-getting-started)
- [Kaggle API 文档](https://github.com/Kaggle/kaggle-api)
- [提交文件格式说明](https://www.kaggle.com/competitions/nlp-getting-started/data)




