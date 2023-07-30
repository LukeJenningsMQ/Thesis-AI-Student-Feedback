
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, logging,AutoModelForTokenClassification,BertForTokenClassification, TokenClassificationPipeline

logging.set_verbosity_warning()
tokenizerBERT = AutoTokenizer.from_pretrained("QCRI/bert-base-multilingual-cased-pos-english")
modelBERT = AutoModelForTokenClassification.from_pretrained('QCRI/bert-base-multilingual-cased-pos-english')
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

def NamedEntityRecognition(text,modelUsed,tokeniserUsed):
    nlp = pipeline("ner", model=modelUsed, tokenizer=tokeniserUsed)
    ner_results = nlp(text)
    print(ner_results)
    return ner_results
def PartOfSpeech(text,modelUsed,tokeniserUsed):
    pipeline = TokenClassificationPipeline(model=modelUsed, tokenizer=tokeniserUsed)
    outputs = pipeline(text)
    print(outputs)
    return outputs
textInput = "Engineers need to follow a proper schedule, and to do this, they should use tools such as a Gantt Chart or a critical path method."
labelsInput = ["Cyber Security", "Engineering", "Art", "Science", "Computing", "Law", "Business","Sport", "Nature"]
sentimentAnalysis(textInput)
zeroShotClassification(textInput, labelsInput)
textGeneration(textInput, modelGPT)
tokens = stringToTokens(textInput, tokenizerBERT)
PartOfSpeech(textInput,modelBERT,tokenizerBERT)