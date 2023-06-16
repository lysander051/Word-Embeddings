#!/usr/bin/env
import spacy, scipy, statistics
import directedlouvain as dl
import sinr.sinr.text.preprocess as ppcs
import networkx as nx
import networkit
import timeit
import pickle
from tqdm import tqdm

from collections import defaultdict

class DirectedLouvain:
    graph = defaultdict(int)
    words_to_ids = dict()
    ids_to_words = dict()
    doc = []
    louvain = None
    matrix = None

    def __init__(self, text="text.txt", pipeline="en_core_web_sm", output_graph="graph.txt", gamma=55, verbose=False, trame=True):
        """
        Uses the directed version of the louvain algorithm to analyse the text file.

        :param text: Path to the text file to analyse.
        :param pipeline: Specify the spacy pipeline to use.
        :param output_graph: name of the graph generated using Spacy
        :param gamma: resolution parameter enabling to control communities' sizes
        :param verbose: enables verbose mode for Directed Louvain's algorithm
        :param trame: enables the whole framework (including spacy), False ignores this part

        See also:
        `Spacy homepage <https://spacy.io/models>`_
        """
        if (not(trame)):
            print("loading data...")
            self.matrix, self.words_to_ids = self.load_data()
            self.graph = nx.from_scipy_sparse_array(self.matrix, create_using=nx.DiGraph)
            print("done.")

        else:
            print("loading spacy pipeline...")
            nlp = spacy.load(pipeline, disable=["ner"])

            if isinstance(text, str):
                text_str = self._read_text(text)
            else:
                parsed_text = ppcs.extract_text(sys.argv[1], lemmatize=True, lower_words=True, number=False, punct=False, en=True, min_freq=20, alpha=True, min_length_word=1)
                text_str = self._read_list(text)

            print("text parsing...")
            for i,sentence in enumerate(tqdm(nlp.pipe(text_str), total=len(text_str))):
                self.doc.append(sentence)

            # make and write the graph inside the graph.txt file and generate a dictionary of words to node
            print("building graph...")
            self._graph_words_to_ids()
            print("done.")
            self._write_graph()

        # computing communities
        print("community detection...") 
        start = timeit.default_timer()
        self.louvain = dl.Community(output_graph, weighted=True, gamma=gamma)
        self.louvain.run(verbose)
        stop = timeit.default_timer()
        dict_communities = self.louvain.last_level()
        word_communities = self._community_of_words(dict_communities)

        communities = [ list() for _ in range(max(dict_communities.values())+1) ]
        for node, community in dict_communities.items():
            communities[community].append(node)

        sizes_communities = [ len(community) for community in communities ]
        print(len(self.words_to_ids),sum(sizes_communities))

        print("Average community size: {}".format(len(self.words_to_ids) / len(word_communities)))
        print("Average community size: {}".format(statistics.mean(sizes_communities)))
        print("Standard deviation for community size: {}".format(statistics.stdev(sizes_communities)))
        print('Time for community detection: ', stop - start)
        print("modularity: " + str(self.louvain.modularity()))
        G=nx.read_weighted_edgelist("graph.txt", nodetype=int, create_using=nx.DiGraph)
        print("graph on {} nodes and {} edges:".format(len(G.nodes),len(G.edges)))
        # save the matrix and dictionary inside the data.pk file

        if(trame):
            self._save_data()

    def get_community(self):
        """
        Creates a networkit Partition (i.e. community) to feed the extract_embeddings function of the SINr library

        :return: a networkit type community graph
        """
        communities = self.louvain.last_level()
        partition = networkit.Partition(len(self._get_networkx_graph()))
        for node, community in communities.items():
            partition.addToSubset(community, node)
        return partition

    # TODO --- passer le nom du fichier en argument avec valeur par défaut
    # NOTE --- pourquoi ne pas initialiser les attributs ?
    def load_data(self):
        """
        Loads the data.pk file to return an adjacency matrix of the graph and a dictionary (word to node)

        :return: the graph as a matrix and the dictionary
        """
        dico = dict()
        matrix = list()
        with open("data.pk", "rb") as savefile:
            dico, matrix = pickle.load(savefile)
        return matrix, dico

    # DONE --- passer le nom du fichier en argument avec valeur par défaut
    def _save_data(self,filename="data.pk"):
        """
        Saves the graph as an adjacency matrix and the word-to-node dictionary in the data.pk file
        """
        
        with open(filename, 'wb') as savefile:
            pickle.dump((self.words_to_ids,
                        scipy.sparse._coo.coo_matrix(nx.to_scipy_sparse_array(self._get_networkx_graph()))),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return filename

    # DONE --- passer le nom du fichier en argument avec valeur par défaut
    def _get_networkx_graph(self,filename="graph.txt"):
        """
        Creates a graph under the networkx format

        :return: a networkx graph of the graph
        """
        return nx.read_weighted_edgelist(filename, nodetype=int, create_using=nx.DiGraph)

    def _graph_words_to_ids(self,window=2):
        """
        Transform a doc type into a graph and a word-to-node dictionary
        """
        numbering = 0
        denumbering = defaultdict()
        for sentence in self.doc:
            for token in sentence:
                if token.dep_ != "ROOT":
                    if token.head.text not in self.words_to_ids:
                        self.words_to_ids[token.head.text] = numbering
                        denumbering[numbering] = token.head.text
                        numbering += 1
                    if token.text not in self.words_to_ids:
                        self.words_to_ids[token.text] = numbering
                        denumbering[numbering] = token.text
                        numbering += 1

                elif token.text not in self.words_to_ids:
                    self.words_to_ids[token.text] = numbering
                    denumbering[numbering] = token.text
                    numbering += 1

            for token in sentence:
                ct = token
                i = 0
                while(i<window):
                    self.graph[(self.words_to_ids[ct.head.text], self.words_to_ids[token.text])] += 1
                    ct = ct.head
                    i += 1
                    if ct.dep_ == "ROOT":
                        break

    def _graph_words_to_ids_bis(self, window=2):
        """
        Transform a doc type into a graph and a word-to-node dictionary
        """
        numbering = 0
        for i,sentence in enumerate(self.doc):
            G = nx.MultiDiGraph()
            localnumbering = { i: str(w) for i,w in enumerate(sentence) }
            for index,token in enumerate(sentence):
                if token.dep_ != "ROOT":
                    if token.head.text not in self.words_to_ids:
                        self.words_to_ids[token.head.text] = numbering
                        numbering += 1
                    if token.text not in self.words_to_ids:
                        self.words_to_ids[token.text] = numbering
                        numbering += 1

                if i==0:
                    G.add_edge(localdict[token.head.text],localdict[token.text])
                    #self.graph[(self.words_to_ids[token.head.text], self.words_to_ids[token.text])] += 1

            if i == 0:
                for e in list(nx.bfs_edges(G, source=localdict["cause"], depth_limit=window)):
                    print(self.words_to_ids[localnumbering[localdict["cause"]]],self.words_to_ids[localnumbering[e[1]]])
                    self.graph[(self.words_to_ids[localnumbering[localdict["cause"]]], self.words_to_ids[localnumbering[e[1]]])] += 1
                print(self.graph)

                #print(G.nodes,token.head.text,token.text)

            '''for source in G.nodes:
                for e in list(nx.bfs_edges(G, source=source,depth_limit=window)):
                    self.graph[(source, e[1])] += 1'''


    def _community_of_words(self, community):
        """
        Uses the words_to_ids dictionary to return a dictionary of words to community with the community parameter.

        :return: a dictionary of community to words
        """
        correspondence = dict()
        for word, index in self.words_to_ids.items():
            comm = community[index]
            correspondence.setdefault(comm, []).append(word)
        return correspondence

    def _read_text(self, filename):
        """
        Reads a .txt file and return a string with its content

        :return: A string of the file
        """
        with open(filename, 'r') as file:
            text = file.read()
        text = text.lower()
        file.close()
        return text

    # NOTE --- attention str est un type en python, il faut changer le nom de la variable
    def _read_list(self, listToRead):
        """
        Reads a .txt file and return a string with its content

        :return: A string of the file
        """
        string = []
        for sentence in listToRead:
            tmp = " ".join(sentence)
            string.append(tmp)
        return string

    # TODO --- passer le nom du fichier en argument avec valeur par défaut
    def _write_graph(self):
        """
        Exports a graph as a .txt format file
        """
        with open("graph.txt", "w") as file:
            for head, tail in self.graph:
                file.write(str(head) + " " + str(tail) + " " + str(self.graph[(head, tail)]) + "\n")


def _parseArgs():
    """
    Parses arguments when this library is used as a main.

    :return: A list of args is returned
    """
    import argparse
    parser = argparse.ArgumentParser(prog='directed_louvain.py',
                                     description='From a text, return the community of each words')
    parser.add_argument('-f', type=str, nargs=1, required=False,
                        help='The path to the name of the file to read')
    parser.add_argument('-p', type=str, nargs=1, required=False,
                        help='The spacy pipeline to use')
    parser.add_argument('-g', type=float, nargs=1, required=False,
                        help='The gamma to use to build smaller or bigger community')
    parser.add_argument("-v", action="store_true", help="increase output verbosity")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parseArgs()
    filename = args.f[0] if args.f else "text.txt"
    pipeline = args.p[0] if args.p else "en_core_web_sm"
    gamma    = args.g[0] if args.g else 55
    verbose  = args.v
    DirectedLouvain(text=filename, pipeline=pipeline, gamma=gamma, verbose=verbose)
