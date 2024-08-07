from transformers import pipeline

# More models in the model hub.
# model_name = "openai/clip-vit-large-patch14-336"
model_name = "openai/clip-vit-base-patch16"

classifier = pipeline("zero-shot-image-classification", model=model_name)

image_to_classify = "./asset/cat.jpg"
labels_for_classification = ["cat", "dog", "lion", "rabbit",
                             "cat or dog",
                             "lion and cheetah",
                             "rabbit and lion"]

# labels_for_classification = ["cat", "dog", "lion", "rabbit"]

scores = classifier(image_to_classify,
                    candidate_labels=labels_for_classification)

print(scores)
