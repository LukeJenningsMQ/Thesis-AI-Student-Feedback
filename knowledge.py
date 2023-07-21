import torch
import transformers
import networkx as nx
import numpy as np

# Load the BERT model.
model = transformers.AutoModel.from_pretrained("bert-base-uncased")

# Get the tokenizer.
tokenizer = model.config.get_tokenizer()

# Define the texts.
texts = ["cat", "dog", "mouse", "bird", "fish"]

# Convert the texts to a list of integers.
text_ids = [tokenizer(text).input_ids for text in texts]

# Convert the list of integers to a tensor.
tensor_texts = torch.tensor(text_ids)

# Extract the embeddings for the entities in your knowledge domain.
embeddings = model(tensor_texts).last_hidden_state

# Create a graph.
graph = nx.Graph()

# Add the entities to the graph.
for word, embedding in zip(texts, embeddings):
    graph.add_node(word)

# Compute the similarity between the embeddings.
similarity_scores = []
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        similarity_scores.append(np.cosine_similarity(embeddings[i], embeddings[j]))

# Use the similarity scores to create the edges in the graph.
for i in range(len(similarity_scores)):
    if similarity_scores[i] > 0.5:
        graph.add_edge(i, j, weight=similarity_scores[i])

# Plot the graph.
nx.draw(graph, with_labels=True)