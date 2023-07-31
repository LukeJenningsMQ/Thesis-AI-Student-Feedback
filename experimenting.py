import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import RobertaTokenizer, BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, logging,AutoModelForTokenClassification,BertForTokenClassification, TokenClassificationPipeline

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
    input = tokeniser.tokenize(text)
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

def similarWords(text, modelUsed, tokeniserUsed,word1Index,word2Index):
    tokens = stringToTokens(text,tokeniserUsed)
    idsForTokens = tokeniserUsed.convert_tokens_to_ids(tokens)
    token_tensor = torch.tensor([idsForTokens])
    modelUsed.eval()
    similarities = list()
    with torch.no_grad():
        output = modelUsed(token_tensor)
        print(output)
        for i in range(1, 13):
           last_hidden_state = output[2][i]
           print(last_hidden_state)
           word_embed_1 = last_hidden_state
           compare1 = word_embed_1[0][word1Index].reshape(1,-1)
           compare2 = word_embed_1[0][word2Index].reshape(1,-1)
           similarity = cosine_similarity(compare1,compare2)
           similarities.append(similarity[0][0])
    return similarities

def plotSimilarities(similarities):
    plt.style.use("Solarize_Light2")
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(range(1,13), similarities, linestyle= "solid")
    plt.title("word embeddings similarities curve from layer 1 to 12")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")



#tokenizerBERT = BertTokenizer.from_pretrained("QCRI/bert-base-multilingual-cased-pos-english")
#modelBERT = BertModel.from_pretrained('QCRI/bert-base-multilingual-cased-pos-english', output_hidden_states = True)
tokenizerBERT = BertTokenizer.from_pretrained("bert-base-uncased")
modelBERT = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
#modelGPT = "distilgpt2"
textInput = "Engineers need to follow a proper schedule and to do this, they should use tools such as a Gantt Chart or a critical path method"
#labelsInput = ["Cyber Security", "Engineering", "Art", "Science", "Computing", "Law", "Business","Sport", "Nature"]
#sentimentAnalysis(textInput)
#zeroShotClassification(textInput, labelsInput)
#textGeneration(textInput, modelGPT)
#tokens = stringToTokens(textInput, tokenizerBERT)
#PartOfSpeech(textInput,modelBERT,tokenizerBERT)
similar = similarWords(textInput,modelBERT,tokenizerBERT,0,6)
plotSimilarities(similar)


