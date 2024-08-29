from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline


# BART的多語言版本是mBART -> https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)


article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
en_text = """
Following the death of the Danish butter magnate Lars Emil Bruun in 1923, his will had a curious order: His vast accumulation of coins, notes and medals, amassed over more than six decades, should be held as an emergency reserve for Denmark’s national collection in case it were ever destroyed. After a century, if all was well, his own cache could finally be sold to benefit his descendants.
Next month, just under a year since the 100-year-old order expired, the first set of coins from Bruun’s personal 20,000-piece collection will go up for auction. It will take several sales to empty Bruun’s coffers, but once completed, it will be the most expensive international coin collection ever sold, according to Stack’s Bowers, the rare coin dealer and auction house hosting the sales. The L.E. Bruun Collection has been insured for 500 million Danish kroner, or around $72.5 million. The auction house describes it as the most valuable collection of world coins to ever come to market.
Where the numismatist’s collection has resided over the past century had been something of a mystery, its location known to few. But Bruun believed that hiding his treasure was for a noble cause; following the destruction he saw of World War I, he feared the Royal Danish Coin and Medal Collection could one day face bombing or looting, according to the auction house.
Bruun began collecting currency as a child in 1859, when his uncle died and named him among the recipients of some of his coins, according to the sales catalog. The son of innkeepers and landowners, he learned in his 20s that his family inheritance had been squandered and he was saddled in debt. He began his own business in butter with a loan, eventually earning a fortune from sales and exports. With his wealth, he became a prolific coin collector, and was a founding member of the Danish Numismatic Society in 1885.
"""
zh_text = """
這裡是你需要生成摘要的長文。這應該是一個詳細的段落或多個段落，你希望將其摘要。
"""


# # translate Hindi to French
# tokenizer.src_lang = "hi_IN"
# encoded_hi = tokenizer(article_hi, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_hi,
#     forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
# )
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# # translate Arabic to English
# tokenizer.src_lang = "ar_AR"
# encoded_ar = tokenizer(article_ar, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_ar,
#     forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
# )
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # => "The Secretary-General of the United Nations says there is no military solution in Syria."


# translate English to zh
tokenizer.src_lang = "en_XX"
encoded_ar = tokenizer(en_text, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
)
r = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Summary:", r)
