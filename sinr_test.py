import sinr.graph_embeddings as ge
import directed_louvain as dl
import sinr.text.preprocess as ppcs

import nltk
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.corpus import gutenberg

louvain = dl.DirectedLouvain(ppcs.extract_text("testbnc.vrt", lemmatize=False, min_freq=5, en=True), "en_core_web_lg")

# creating the SINr object from matrix and dico
sinr = ge.SINr.load_from_adjacency_matrix(*louvain.load_data())

# computing communities using Directed Louvain
communities = louvain.get_community()

# computing embeddings
sinr.extract_embeddings(communities)
sinr_vectors = ge.ModelBuilder(sinr, "corpus", n_jobs=8, n_neighbors=5).with_embeddings_nr().with_vocabulary().with_communities().build()

print("\nType your word to search its neighbors or search for an empty word to exit:")
while True:
    try:
        word = input()
        if word == "":
            break
        print(sinr_vectors.most_similar(word))
    except:
        print("Couldn't find this word")
