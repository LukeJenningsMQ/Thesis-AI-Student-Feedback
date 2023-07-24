
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, logging


logging.set_verbosity_warning()
tokenizerRoberta = AutoTokenizer.from_pretrained('bert-base-cased')
modelRoberta = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
modelGPT = "distilgpt2"

def sentimentAnalysis(text):
    classifier = pipeline("sentiment-analysis")
    predictions = classifier(text)
    print(predictions)
    
def zeroShotClassification(text, labels):
    classifier = pipeline("zero-shot-classification")
    res = classifier(text, candidate_labels=labels)
    print(res)
    
def textGeneration(text, modelName):
    generator = pipeline("text-generation", model=modelName)
    res = generator(text, max_length=30,num_return_sequences=1)
    print(res)
    
def stringToTokens(text, tokeniser):
    input = tokeniser(text)
    print(input)
    return input
    
textInput = "i love trees. We can grow them and make the world a better place as i provides oxygen for earth. Flower seeds grass ocean"
labelsInput = ["Cyber Security", "Engineering", "Art", "Science", "Computing", "Law", "Business","Sport", "Nature"]
sentimentAnalysis(textInput)
zeroShotClassification(textInput, labelsInput)
textGeneration(textInput, modelGPT)
stringToTokens(textInput, tokenizerRoberta)