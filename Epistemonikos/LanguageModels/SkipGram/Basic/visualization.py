from open_documents import PaperReader
import pickle
import json
import collections
import numpy as np

with open("SkipGram/embedding", "rb") as embed_serialized:
    final_embeddings = pickle.load(embed_serialized)
print(final_embeddings.shape, final_embeddings[0].shape)

with open("SkipGram/documents_array.json", "r") as json_file:
    loaded = json.load(json_file)

with open("SkipGram/count", "rb") as count_file:
    count = pickle.load(count_file)

with open("SkipGram/reverse_dictionary", "rb") as reverse_dictionary_file:
    reverse_dictionary = pickle.load(reverse_dictionary_file)



# Step 1: Ponderation vector per document type.
print("step 1")


document_types = ["systematic-review",
                  "structured-summary-of-systematic-review",
                  "primary-study",
                  "overview",
                  "structured-summary-of-primary-study"]

document_vectors = list()
reader = PaperReader(loaded)

for type_ in document_types:
    reader.remove_all_filters()
    reader.apply_filter(type_)
    reader.generate_words_list()
    type_words = reader.words
    type_count = collections.Counter(type_words)
    vector = []
    for word, _ in count:
        if word in type_count:
            vector.append(type_count[word])
            del type_count[word]
        else:
            vector.append(0)
    freq = np.array(vector, ndmin=2)
    nwords = sum(vector)
    rel_freq = freq / nwords
    document_vectors.append(rel_freq)

document_embeds = np.asarray([np.dot(vector, final_embeddings) for vector in document_vectors])
document_embeds = np.asarray([matrix for wrapped_matrix in document_embeds for matrix in wrapped_matrix])




# Step 2: Visualize the embeddings.
print("step 2")

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    if label in document_types:
        color = "r"
        size = 70
    else:
        color = "b"
        size = 7
    plt.scatter(x, y, c = color, s = size)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 1000
  to_plot_list = []
  for i in range(plot_only):
      to_plot_list.append(final_embeddings[i])
  for document in document_embeds:
      to_plot_list.append(document)
  to_plot = np.asarray(to_plot_list)
  print(to_plot)
  print(to_plot.shape)

  low_dim_embs = tsne.fit_transform(to_plot)
  labels = [reverse_dictionary[i] for i in range(plot_only)] + document_types
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
