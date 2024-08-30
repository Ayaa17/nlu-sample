from transformers import T5Tokenizer, T5ForConditionalGeneration

# 定义模型名称
model_name = "t5-base"

# 加载 T5 的 tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 示例文本
text = """
T5 is a model developed by Google that is designed to convert various NLP tasks into a text-to-text format. It is versatile and can be used for tasks like translation, summarization, and question answering. T5 is pre-trained on a large corpus and fine-tuned for specific tasks.
"""

en_text = """
Following the death of the Danish butter magnate Lars Emil Bruun in 1923, his will had a curious order: His vast accumulation of coins, notes and medals, amassed over more than six decades, should be held as an emergency reserve for Denmark’s national collection in case it were ever destroyed. After a century, if all was well, his own cache could finally be sold to benefit his descendants.
Next month, just under a year since the 100-year-old order expired, the first set of coins from Bruun’s personal 20,000-piece collection will go up for auction. It will take several sales to empty Bruun’s coffers, but once completed, it will be the most expensive international coin collection ever sold, according to Stack’s Bowers, the rare coin dealer and auction house hosting the sales. The L.E. Bruun Collection has been insured for 500 million Danish kroner, or around $72.5 million. The auction house describes it as the most valuable collection of world coins to ever come to market.
Where the numismatist’s collection has resided over the past century had been something of a mystery, its location known to few. But Bruun believed that hiding his treasure was for a noble cause; following the destruction he saw of World War I, he feared the Royal Danish Coin and Medal Collection could one day face bombing or looting, according to the auction house.
Bruun began collecting currency as a child in 1859, when his uncle died and named him among the recipients of some of his coins, according to the sales catalog. The son of innkeepers and landowners, he learned in his 20s that his family inheritance had been squandered and he was saddled in debt. He began his own business in butter with a loan, eventually earning a fortune from sales and exports. With his wealth, he became a prolific coin collector, and was a founding member of the Danish Numismatic Society in 1885.
"""

# 将任务类型附加到输入文本
input_text = "summarize: " + en_text

# 编码输入文本
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# 生成摘要
summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

# 解码并输出摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\nSummary:")
print(summary)
