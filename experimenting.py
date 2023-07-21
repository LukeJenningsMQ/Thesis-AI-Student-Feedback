
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer

tokenizerRoberta = AutoTokenizer.from_pretrained('roberta-base')
modelRoberta = AutoModelForSequenceClassification.from_pretrained('roberta-base')
modelGPT = "distilgpt2"

def sentimentAnalysis(text, modelName):
    classifier = pipeline("sentiment-analysis", model=modelName, tokenizer=tokenizerRoberta)
    predictions = classifier(text)

    # Get the predicted class.
    predicted_class = predictions["label"]

    # Print the predicted class.
    print(f"The predicted class is: {predicted_class}.")

    # Get the confidence score.
    score = predictions["score"]

    # Print the confidence score.
    print(f"The confidence score is: {score}.")
    
def zeroShotClassification(text, labels, modelName):
    classifier = pipeline("zero-shot-classification", model=modelName, tokenizer=tokenizerRoberta)
    res = classifier(text, candidate_labels=labels)
    print(res)
    
def textGeneration(text, modelName):
    generator = pipeline("text-generation", model=modelName)
    res = generator(text, max_length=30,num_return_sequences=1)
    print(res)
    
    
textInput = "i love trees. We can grow them and make the world a better place as i provides oxygen for earth. Flower seeds grass ocean"
labelsInput = ["Cyber Security", "Engineering", "Art", "Science", "Computing", "Law", "Business","Sport", "Nature"]
sentimentAnalysis(textInput, modelRoberta)
zeroShotClassification(textInput, labelsInput, modelRoberta)
textGeneration(textInput, modelGPT)