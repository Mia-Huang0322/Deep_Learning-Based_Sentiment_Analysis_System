from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn as nn

# 1) 加载 IMDB
dataset = load_dataset("imdb",cache_dir='/mnt/data/hf_cache/datasets')

# 2) 分词器改为 roberta-large
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

def tokenizer_function(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=256,          # 可根据文本长度分布适当调整
    )

# 3) 分词
tokenized = dataset.map(tokenizer_function, batched=True, remove_columns=["text"])

# 4) 划分训练/验证
splits = tokenized["train"].train_test_split(test_size=0.2)
train_dataset = splits["train"]
eval_dataset  = splits["test"]

# 5) 构建带“多层 Dropout”分类头的自定义模型
from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaForSeqClsMultiDropout(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        # 获取 RoBERTa 的输出，return_dict=True 保证返回的是字典
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # 取 [CLS] 的表示作为句子向量（第 0 个 token）
        hidden_state = outputs.last_hidden_state          # shape: [batch, seq_len, hidden_dim]
        pooled_output = hidden_state[:, 0]                # shape: [batch, hidden_dim]

        # 多层 Dropout + MLP 分类器
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 加载带自定义 head 的模型
config = RobertaConfig.from_pretrained(
    "roberta-large",
    num_labels=2,
    problem_type="single_label_classification",
    hidden_dropout_prob=0.3,        # roberta 自身层的 dropout
    attention_probs_dropout_prob=0.3
)
model = RobertaForSeqClsMultiDropout.from_pretrained("roberta-large", config=config)

# 6+7+13) 训练参数：梯度累积、warmup、调度、早停、保存最优
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,      # 相当于 bs=16
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,                   # 预热步数
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# 自定义指标函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 8) 启动 Trainer，添加 EarlyStoppingCallback（patience=1）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# 9) 开始训练
trainer.train()

# 10) 保存最终模型（Trainer 在每 epoch 已保存最优到 output_dir）
trainer.save_model("./final_roberta")

# 11) 在验证集上评估
metrics = trainer.evaluate()
print(metrics)


import pandas as pd
import matplotlib.pyplot as plt

# 把 log_history 转成 DataFrame
logs = trainer.state.log_history
df = pd.DataFrame(logs)

# 只保留含训练 loss 的行
df_train = df[['epoch', 'loss']].dropna()

# 只保留含验证指标的行
df_eval = df[['epoch', 'eval_loss', 'eval_accuracy', 'eval_f1']].dropna()

# 画训练 loss 和验证 loss 对比
plt.figure()
plt.plot(df_train['epoch'], df_train['loss'], label='train_loss')
plt.plot(df_eval['epoch'], df_eval['eval_loss'], label='eval_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# 画验证 accuracy 和 f1
plt.figure()
plt.plot(df_eval['epoch'], df_eval['eval_accuracy'], label='eval_accuracy')
plt.plot(df_eval['epoch'], df_eval['eval_f1'], label='eval_f1')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend()
plt.show()
