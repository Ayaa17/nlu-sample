from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 定义模型名称（GPT-2 基础模型）
model_name = "gpt2"

# 加载模型和 tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 示例文本
input_text = """
T5 is a model developed by Google that is designed to convert various NLP tasks into a text-to-text format. It is versatile and can be used for tasks like translation, summarization, and question answering. T5 is pre-trained on a large corpus and fine-tuned for specific tasks.
"""

# 为生成摘要设置提示
prompt = "Summarize the following text:\n\n" + input_text + "\n\nSummary:"

# 将输入文本编码
inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)

# 生成摘要
output = model.generate(
    inputs,
    max_length=150,  # 设置生成文本的最大长度
    num_beams=4,     # 使用 beam search 来提高生成质量
    no_repeat_ngram_size=2,  # 避免重复生成相同的短语
    early_stopping=True  # 提前停止生成
)

# 解码生成的摘要
summary = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("Summary:")
print(summary)
