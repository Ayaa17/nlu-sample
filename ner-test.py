import torch
from transformers import BertTokenizer, BertModel

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输入句子
sentence = "OpenAI's GPT-3 is a powerful language model."
sentence = "I want to find information about AI."
sentence = "I want to find dog and cat."
sentence = "Apple is looking at buying U.K. startup for $1 billion."
sentence = "I have a dog and a cat."

# 对句子进行分词和编码
inputs = tokenizer(sentence, return_tensors='pt')

# 获取 BERT 模型的输出
with torch.no_grad():
    outputs = model(**inputs)

# 提取最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state

# 获取每个词的嵌入向量
word_embeddings = last_hidden_states[0]

# 计算每个词嵌入向量的范数
word_norms = torch.norm(word_embeddings, dim=1)

# 选择范数最大的几个词作为关键字
top_k = 5  # 选择前5个关键字
top_k_indices = torch.topk(word_norms, top_k).indices

# 将索引转换回词汇
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
keywords = [tokens[idx] for idx in top_k_indices]

# 去掉特殊字符，并确保只返回有意义的词
keywords = [keyword for keyword in keywords if keyword not in tokenizer.all_special_tokens]

print("Extracted Keywords:", keywords)

