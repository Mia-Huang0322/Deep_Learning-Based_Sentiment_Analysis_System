import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("./final_model")
tokenizer = BertTokenizer.from_pretrained("./final_model")

text = "Although the weather is bad today, I am still very happy."


# 3. 使用 tokenizer 将文本转换为模型输入
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 4. 进行推理（预测）
with torch.no_grad():
    outputs = model(**inputs)

# 5. 获取预测结果（logits）
logits = outputs.logits

# 6. 使用 softmax 获取概率分布
softmax = torch.nn.Softmax(dim=-1)
probabilities = softmax(logits)

# 7. 打印输出类别和对应概率
predicted_class = torch.argmax(probabilities, dim=-1).item()
print(f"Predicted class: {predicted_class}, Probabilities: {probabilities}")

# 8. 根据模型设置，通常 0 是消极，1 是积极（假设这是二分类模型）
if predicted_class == 0:
    print("The sentiment is Negative.")
else:
    print("The sentiment is Positive.")

