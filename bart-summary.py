from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# 加載預訓練的BART模型和tokenizer

# BART模型，特別針對CNN/DailyMail數據集進行了調優
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 定義一個文本摘要生成pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# 要進行摘要的文章
# source: https://edition.cnn.com/2024/08/26/style/denmark-le-bruun-coin-collection-auction/index.html
en_text = """
Following the death of the Danish butter magnate Lars Emil Bruun in 1923, his will had a curious order: His vast accumulation of coins, notes and medals, amassed over more than six decades, should be held as an emergency reserve for Denmark’s national collection in case it were ever destroyed. After a century, if all was well, his own cache could finally be sold to benefit his descendants.
Next month, just under a year since the 100-year-old order expired, the first set of coins from Bruun’s personal 20,000-piece collection will go up for auction. It will take several sales to empty Bruun’s coffers, but once completed, it will be the most expensive international coin collection ever sold, according to Stack’s Bowers, the rare coin dealer and auction house hosting the sales. The L.E. Bruun Collection has been insured for 500 million Danish kroner, or around $72.5 million. The auction house describes it as the most valuable collection of world coins to ever come to market.
Where the numismatist’s collection has resided over the past century had been something of a mystery, its location known to few. But Bruun believed that hiding his treasure was for a noble cause; following the destruction he saw of World War I, he feared the Royal Danish Coin and Medal Collection could one day face bombing or looting, according to the auction house.
Bruun began collecting currency as a child in 1859, when his uncle died and named him among the recipients of some of his coins, according to the sales catalog. The son of innkeepers and landowners, he learned in his 20s that his family inheritance had been squandered and he was saddled in debt. He began his own business in butter with a loan, eventually earning a fortune from sales and exports. With his wealth, he became a prolific coin collector, and was a founding member of the Danish Numismatic Society in 1885.
"""

# 生成摘要
summary = summarizer(en_text, max_length=130, min_length=30, do_sample=False)

# 輸出摘要結果
print("Summary:", summary[0]['summary_text'])
