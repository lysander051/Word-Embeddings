import sinr.graph_embeddings as ge
import directed_louvain as dl

import nltk
nltk.download("gutenberg")
from nltk.corpus import gutenberg


louvain = dl.DirectedLouvain(gutenberg.sents('shakespeare-macbeth.txt'))

# creating the SINr object from matrix and dico
sinr = ge.SINr.load_from_adjacency_matrix(*louvain.load_data())

# computing communities using Directed Louvain
communities = louvain.get_community()

# computing embeddings
sinr.extract_embeddings(communities)
sinr_vectors = ge.ModelBuilder(sinr, "harry", n_jobs=8, n_neighbors=5).with_embeddings_nr().with_vocabulary().with_communities().build()

while True:
    try:
        print(sinr_vectors.most_similar(input()))
    except:
        print("mot non présent")
