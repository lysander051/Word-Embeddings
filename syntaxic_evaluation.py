from nltk.corpus import reuters
import sinr.sinr.graph_embeddings as ge
import sinr.sinr.text.preprocess as ppcs
from sinr.sinr.text.pmi import pmi_filter
import directed_louvain as dl
import sys

if(sys.argv[1].split(".")[-1] == "pk"):
    louvain = dl.DirectedLouvain(trame=False,gamma=50)
else:
    louvain = dl.DirectedLouvain(ppcs.extract_text(sys.argv[1], lemmatize=True, lower_words=True, number=False, punct=False, en=True, min_freq=20, alpha=True, min_length_word=1), gamma=50)

# creating the SINr object from matrix and dico
sinr = ge.SINr.load_from_adjacency_matrix(*louvain.load_data())

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
#sinr_vectors.light_model_save() #Cette fonction sauve pas l'objet model, mais directement le dictionnaire mot -> array pour que ce soit évaluable

#sinr_vectors_new = ge.SINrVectors("corpus_light") #déclaration de l'objet sinr avec le nom du .pk du modele
#sinr_vectors_new.load()

'''import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RG65, fetch_MTurk, fetch_RW
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity
import os

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

models = [ "corpus_light.pk" ]
###datasets potentiels en similarité

tasks = {
    "MEN": fetch_MEN(),                          #standard
    "WS353": fetch_WS353(),                      #standard
    "WS353R": fetch_WS353(which="relatedness"),
    "WS353S": fetch_WS353(which="similarity"),   #idéalement les embeddings syntaxiques sont meilleures sur WS353S que WS353R
    "SIMLEX999": fetch_SimLex999(),             #idéalement également les embeddings syntaxiques sont meilleurs que les embeddings normaux sur celui là
    "RG65" : fetch_RG65(),                      #mauvais dataset pas significatif
    "MTURK" : fetch_MTurk(),                    #standard
    "RW" : fetch_RW()                             #dataset sur les mors rares, vous pouvez regarder mais c'est anecdotique
}

dataw = []

for m in models :
    model = load_embedding(m,format='dict',normalize=False)
    results = []

    # Calculate results using helpewith open('/lium/raid01_c/sguillot/Datasets/BLESS_cosined.csv', 'w') as file :r function
    for name, data in iteritems(tasks):
        print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(model, data.X, data.y)))
        results.append(evaluate_similarity(model, data.X, data.y))
    dataw.append(results)

import csv
with open('syntaxic_similarity.csv', 'w') as file :
    writer = csv.writer(file, delimiter=';')
    for d in dataw :
        writer.writerow(d)'''
