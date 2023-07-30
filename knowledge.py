import transformers
from transformers import RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoTokenizer, logging, AutoModel
import sklearn.manifold
import matplotlib.pyplot as plt
import torch


# Load the BERT model.
model = transformers.AutoModel.from_pretrained("bert-base-uncased")

# Create a tokenizer.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the texts.
texts = ["This is a text about BERT.", "This is a text about RoBERTa."]
tokens = [tokenizer(text) for text in texts]

# Convert the tokens to a list of integers.
token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]

# Convert the tokens to a tensor of integers.
tokens_tensor = torch.tensor(token_ids, dtype=torch.int64)

# Extract the embeddings for the tokens.
embeddings = model.embeddings.word_embeddings(tokens_tensor)
embeddings = embeddings.detach()
embeddings = embeddings.view(-1, 2)

# Create a knowledge map from the embeddings.
tsne = sklearn.manifold.TSNE(n_components=2, perplexity=1)
embeddings_tsne = tsne.fit_transform(embeddings)
# Convert the token IDs to tokens.
tokens = tokenizer.convert_ids_to_tokens(embeddings_tsne)

# Iterate through the list of tokens and print the tokens and their corresponding coordinates.
for i, token in enumerate(tokens):
    print(f"{token}: {embeddings_tsne[i]}")
# Visualize the knowledge map.

plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
plt.show()