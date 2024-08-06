from transformers import pipeline

def classify(sentences, candidate_labels):
    classifier = pipeline("zero-shot-classification")
    for sentence in sentences:
        result = classifier(sentence, candidate_labels)
        print(f"({result['sequence']})->{result['labels'][0]}")


if __name__ == '__main__':
    sentences = [
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
        # "The cat jumped over the lazy dog.",
        # "She sells seashells by the seashore.",
        # "An apple a day keeps the doctor away.",
        # "He drove his car to the mountain peak.",
        # "The quick brown fox ran swiftly through the forest.",
        # "A stitch in time saves nine.",
        # "They decided to paint the house blue.",
        # "The spaceship landed on the red planet.",
    ]

    # candidate_labels = ["open app", "close app"]
    candidate_labels = ["close", "open"]


    classify(sentences, candidate_labels)


