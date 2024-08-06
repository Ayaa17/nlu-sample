import math

import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf


def classify(sentences, candidate_labels):
    """
    將每個句子轉換為嵌入向量。
    計算每個句子的嵌入向量與指令嵌入向量之間的餘弦相似度。
    如果最大的相似度超過0.4，則判斷該句子屬於相應的指令；否則，判斷為無關。
    """

    # 加载Universal Sentence Encoder模型
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(model_url)

    instruction_embeddings = embed(candidate_labels)

    for sentence in sentences:
        text_embedding = embed([sentence])

        similarities = []
        for instr_emb in instruction_embeddings:
            instr_emb_2d = tf.reshape(instr_emb, (1, -1))
            similarities.append(cosine_similarity(text_embedding, instr_emb_2d)[0][0])

        max_similarity = max(similarities)
        if max_similarity > 0.4:
            result = candidate_labels[np.argmax(similarities)]
            print(f"({sentence}) -> {result}")
        else:
            print(f"({sentence}) -> irrelevant: {max_similarity}")


def classify2(sentences, candidate_labels):
    """
    將每個句子轉換為嵌入向量並進行正規化。
    計算每個句子的正規化嵌入向量與指令正規化嵌入向量之間的內積。
    將內積值裁剪到[-1.0, 1.0]之間，並進一步計算分數。
    如果最大的分數超過0.6，則判斷該句子屬於相應的指令；否則，判斷為無關。
    """
    # 加载Universal Sentence Encoder模型
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(model_url)

    labels_encode = tf.nn.l2_normalize(embed(candidate_labels), axis=1)

    for sentence in sentences:
        sts_encode2 = tf.nn.l2_normalize(embed([sentence]), axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(labels_encode, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        max_similarity = max(scores)
        result = candidate_labels[np.argmax(scores)]

        if max_similarity > 0.6:
            print(f"({sentence}) -> {result}")
        else:
            print(f"({sentence}) -> irrelevant: {result} :{max_similarity}")


if __name__ == '__main__':
    sentences = [
        # else
        "Please launch the weather app.",
        "Open the music player.",
        "Can you access the fitness tracker?",
        "Start the calculator app.",
        "Activate the navigation app.",
        "Could you open the news reader?",
        "Open the camera app.",
        "Can you start the note-taking software?",
        "Open the calendar app.",
        "Please close the flashlight app.",
        "Please shut down the email app.",
        "Can you close the game application?",
        "Turn off the alarm app.",
        "Disable the translation app.",
        "Exit the social media app.",

        #  unrelated
        "The cat jumped over the lazy dog.",
        "She sells seashells by the seashore.",
        "An apple a day keeps the doctor away.",
        "He drove his car to the mountain peak.",
        "The quick brown fox ran swiftly through the forest.",
        "A stitch in time saves nine.",
        "They decided to paint the house blue.",
        "The spaceship landed on the red planet.",
    ]

    candidate_labels = ["open app", "close app", "app", "open", "close"]
    # candidate_labels = ["close", "open"]

    # test_text = ["啟動系統"]
    # candidate_labels = ["請啟動系統", "請關閉系統", "請重啟系統"]

    # classify(sentences, candidate_labels)
    classify2(sentences, candidate_labels)
