import sinr.sinr.graph_embeddings as ge
import sys

sinr_vectors = ge.SINrVectors("corpus") #déclaration de l'objet sinr avec le nom du .pk du modele
sinr_vectors.load()

desc = sinr_vectors.get_obj_descriptors(sys.argv[1], topk_dim=5, topk_val=5)
ster = sinr_vectors.get_obj_stereotypes(sys.argv[1], topk_dim=5, topk_val=5)

for d in desc:
    print(list(d.values())[2:])

print()

for s in ster:
    print(list(s.values())[2:])
