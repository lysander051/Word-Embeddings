import sinr.sinr.graph_embeddings as ge
import directed_louvain as dl
import sys

# creating the SINr object from matrix and dico only if graph has been generated
louvain = dl.DirectedLouvain(filename=sys.argv[1],trame=False,gamma=50)
sinr = ge.SINr.load_from_adjacency_matrix(louvain.matrix, louvain.words_to_ids)

# computing communities using Directed Louvain 
communities = louvain.get_community()
sinr.extract_embeddings(communities)

sinr_vectors = ge.InterpretableWordsModelBuilder(sinr, "corpus", n_jobs=8, n_neighbors=5).build()#.with_embeddings_nr().with_vocabulary().build()

print("\nType your word to search its neighbors or search for an empty word to exit:")
while True:
    try:
        word = input()
        if word == "":
            break
        desc = sinr_vectors.get_obj_descriptors(word, topk_dim=5, topk_val=5)
        ster = sinr_vectors.get_obj_stereotypes(word, topk_dim=5, topk_val=5)

        for d in desc:
            print(list(d.values())[2:])

        print()

        for s in ster:
            print(list(s.values())[2:])

    except:
        print("Couldn't find this word")
