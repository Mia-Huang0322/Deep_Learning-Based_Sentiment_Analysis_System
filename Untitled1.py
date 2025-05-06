#%%
from datasets import load_dataset
dataset=load_dataset("imdb")
print(dataset)
#%%
from transformers import BertTokenizer
#%%
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

#%%
def tokenizer_function(ex):
  return tokenizer(ex["text"],truncation=True,padding="max_length")
#%%
tokenizer_datasets=dataset.map(tokenizer_function,batched=True)
#%%
train_t=tokenizer_datasets["train"].train_test_split(test_size=0.2)
train_dataset=train_t['train']
test_dataset=train_t['test']
#%%
from transformers import BertForSequenceClassification,AdamW
#%%
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
#%%
from transformers import Trainer,TrainingArguments
#%%
training_args=TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
#%%
trainer.save_model("./final_model")  # 指定保存的目录

#%%
m=trainer.evaluate()
#%%
m
#%%
prediction=trainer.predict(test_dataset)
#%%
prediction
#%%
test_dataset['label']
#%%
# 应用 softmax 函数来转换 logits 为概率
from scipy.special import softmax
probabilities = softmax(prediction.predictions, axis=-1)

# 获取每个样本预测的类别（即选择概率最高的类别）
predicted_labels = np.argmax(probabilities, axis=-1)

print(predicted_labels)  # 输出预测的类别（0 或 1）
label_ids = prediction.label_ids
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(label_ids, predicted_labels)
print(accuracy)