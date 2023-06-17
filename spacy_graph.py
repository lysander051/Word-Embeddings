import directed_louvain as dl
import sys

louvain = dl.DirectedLouvain(filename=sys.argv[1], gamma=50)

