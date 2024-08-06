from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def classify(sentences, candidate_labels):
    # 加載預訓練的BERT模型和分詞器
    # bert-base-uncased, distilbert-base-uncased, albert-base-v2, huawei-noah/TinyBERT_General_4L_312D
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 定義函數來獲取文本嵌入
    def get_embeddings(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    # 獲取指令的嵌入
    instruction_embeddings = [get_embeddings(instr, tokenizer, model) for instr in candidate_labels]
    print(instruction_embeddings[0].shape)

    for sentence in sentences:
        text_embedding = get_embeddings(sentence, tokenizer, model)
        print(text_embedding.shape)

        similarities = [cosine_similarity(text_embedding, instr_emb)[0][0] for instr_emb in instruction_embeddings]
        max_similarity = max(similarities)
        if max_similarity > 0.6:
            result = candidate_labels[np.argmax(similarities)]
            print(f"({sentence}) -> {result}")

        else:
            print(f"({sentence}) -> irrelevant: {max_similarity}")


if __name__ == '__main__':
    sentences = [

        # # else
        # "Please launch the weather app.",
        # "Open the music player.",
        # "Can you access the fitness tracker?",
        # "Start the calculator app.",
        # "Activate the navigation app.",
        # "Could you open the news reader?",
        # "Open the camera app.",
        # "Can you start the note-taking software?",
        # "Open the calendar app.",
        # "Please close the flashlight app.",
        # "Please shut down the email app.",
        # "Can you close the game application?",
        # "Turn off the alarm app.",
        # "Disable the translation app.",
        # "Exit the social media app.",
        #
        # #  unrelated
        # "The cat jumped over the lazy dog.",
        # "She sells seashells by the seashore.",
        # "An apple a day keeps the doctor away.",
        # "He drove his car to the mountain peak.",
        # "The quick brown fox ran swiftly through the forest.",
        # "A stitch in time saves nine.",
        # "They decided to paint the house blue.",
        # "The spaceship landed on the red planet.",
        #

        # youtube
        # "Could you pull up YouTube for me?",
        # "Please launch YouTube.",
        # "Open YouTube, please.",
        # "Would you mind starting YouTube?",
        # "Can you access YouTube?",
        # "Fire up YouTube.",
        # "Could you bring up YouTube?",
        # "Please get YouTube going.",
        # "Start YouTube for me.",
        # "Turn on YouTube, would you?",
        # "start my favorite youtube",
    ]

    # candidate_labels = ["open InstaShare", "close InstaShare"]
    candidate_labels = ["open youtube", "open browser"]
    # candidate_labels = ["close", "open"]

    instructions = ["請啟動系統", "請關閉系統", "請重啟系統"]
    test_text = "啟動系統"

    classify(sentences, candidate_labels)
