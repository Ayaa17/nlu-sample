from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# 定义模型名称
model_name = "google/pegasus-xsum"

# 加载模型和tokenizer
print("Loading tokenizer...")
tokenizer = PegasusTokenizer.from_pretrained(model_name)

print("Loading model...")
model = PegasusForConditionalGeneration.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")

# 示例文本
text = """
Pegasus is a state-of-the-art summarization model developed by Google Research. It is specifically trained to generate abstractive summaries for long documents. The model is based on the transformer architecture and pre-trained on large-scale datasets.
"""

en_text = """
Following the death of the Danish butter magnate Lars Emil Bruun in 1923, his will had a curious order: His vast accumulation of coins, notes and medals, amassed over more than six decades, should be held as an emergency reserve for Denmark’s national collection in case it were ever destroyed. After a century, if all was well, his own cache could finally be sold to benefit his descendants.
Next month, just under a year since the 100-year-old order expired, the first set of coins from Bruun’s personal 20,000-piece collection will go up for auction. It will take several sales to empty Bruun’s coffers, but once completed, it will be the most expensive international coin collection ever sold, according to Stack’s Bowers, the rare coin dealer and auction house hosting the sales. The L.E. Bruun Collection has been insured for 500 million Danish kroner, or around $72.5 million. The auction house describes it as the most valuable collection of world coins to ever come to market.
Where the numismatist’s collection has resided over the past century had been something of a mystery, its location known to few. But Bruun believed that hiding his treasure was for a noble cause; following the destruction he saw of World War I, he feared the Royal Danish Coin and Medal Collection could one day face bombing or looting, according to the auction house.
Bruun began collecting currency as a child in 1859, when his uncle died and named him among the recipients of some of his coins, according to the sales catalog. The son of innkeepers and landowners, he learned in his 20s that his family inheritance had been squandered and he was saddled in debt. He began his own business in butter with a loan, eventually earning a fortune from sales and exports. With his wealth, he became a prolific coin collector, and was a founding member of the Danish Numismatic Society in 1885.
"""

# 编码输入文本
inputs = tokenizer(en_text, max_length=1024, return_tensors="pt", truncation=True)

# 生成摘要
summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)

# 解码并输出摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\nSummary:")
print(summary)
