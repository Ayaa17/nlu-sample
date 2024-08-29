from transformers import pipeline

# 加載預訓練的 NER 模型
# nlp_ner = pipeline("ner", aggregation_strategy="simple")
nlp_ner = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")



def extract_query(sentence):
    # 使用 NER 模型進行實體識別
    ner_results = nlp_ner(sentence)

    # 定義查找的關鍵詞
    search_keywords = ["search", "find", "look for", "want to find"]
    query_terms = []

    # 遍歷 NER 結果，檢查是否包含查找關鍵詞
    for entity in ner_results:
        print(entity)
        if entity['word'].lower() in search_keywords:
            # 提取查找詞的後續詞
            query_terms.append(entity['start'])

    # 提取查找詞的後續詞
    for idx in query_terms:
        if idx + 1 < len(sentence.split()):
            query_terms.append(sentence.split()[idx + 1])

    return list(set(query_terms))  # 去除重複的查找詞


# 測試句子
sentence = "I want to find information about AI."
# sentence = "I want to find dog and cat."
# sentence = "Apple is looking at buying U.K. startup for $1 billion."
# sentence = "I have a dog and a cat."

result = extract_query(sentence)
print("用戶想要查找的詞:", result)


# # 加載預訓練的填空模型
# fill_mask = pipeline("fill-mask", model="bert-base-uncased")
#
#
# def extract_keywords_with_llm(sentence):
#     # 將句子分割成單詞
#     words = sentence.split()
#     # keywords = set()
#     keywords = []
#
#     # 對每個單詞進行填空測試
#     for word in words:
#         masked_sentence = sentence.replace(word, '[MASK]')
#         predictions = fill_mask(masked_sentence)
#         # 將預測的詞添加到關鍵詞列表
#         # keywords.extend([pred['token_str']] for pred in predictions)
#         # 將預測的詞添加到關鍵詞列表
#         for pred in predictions:
#             # 添加預測的詞，並確保不重複
#             # keywords.add([pred['token_str']])
#             print(pred)
#
#     return set(keywords)  # 去除重複的關鍵詞
#
#
# # 測試句子
# sentence = "I have a dog and a cat."
# keywords = extract_keywords_with_llm(sentence)
# print("提取的關鍵詞:", keywords)
