#!/usr/bin/env
import spacy, scipy
import directedlouvain as dl
import networkx as nx
import networkit
import timeit
import pickle
from tqdm import tqdm

from collections import defaultdict

class DirectedLouvain:
    graph = defaultdict(int)
    reference = defaultdict(int)
    anti_reference = defaultdict(int)
    doc = []
    louvain = None

    def __init__(self, text="text.txt", pipeline="en_core_web_sm", gamma=55, verbose=False):
        """
        Uses the directed version of the louvain algorithm to analyse the text file.

        :param text: Path to the text file to analyse.
        :param pipeline: Specify the spacy pipeline to use.

        See also:
        `Spacy homepage <https://spacy.io/models>`_
        """
        print("loading spacy pipeline...")
        nlp = spacy.load(pipeline, disable=["ner"])

        if isinstance(text, str):
            text_str = self._read_text(text)
        else:
            text_str = self._read_list(text)
        print("text parsing...")
        for i,sentence in enumerate(tqdm(nlp.pipe(text_str), total=len(text_str))):
            self.doc.append(sentence)

        # make and write the graph inside the graph.txt file and generate a dictionary of words to node
        print("building graph...")
        self._graph_reference()
        print("done.")
        self._write_graph()

        # computing communities
        print("community detection...") 
        start = timeit.default_timer()
        self.louvain = dl.Community("graph.txt", weighted=True, gamma=gamma)
        self.louvain.run(verbose)
        stop = timeit.default_timer()
        community = self._community_of_words(self.louvain.last_level(), self.reference)
        print("Average community size: " + str(len(self.reference) / len(community)))
        print('Time for community detection: ', stop - start)
        print("modularity: " + str(self.louvain.modularity()))

        # save the matrix and dictionary inside the data.pk file
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

    # TODO --- passer le nom du fichier en argument avec valeur par défaut
    def _save_data(self):
        """
        Saves the graph as an adjacency matrix and the word-to-node dictionary in the data.pk file
        """
        filename = "data.pk"
        with open(filename, 'wb') as savefile:
            pickle.dump((self.reference,
                        scipy.sparse._coo.coo_matrix(nx.to_scipy_sparse_array(self._get_networkx_graph()))),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return filename

    # TODO --- passer le nom du fichier en argument avec valeur par défaut
    def _get_networkx_graph(self):
        """
        Creates a graph under the networkx format

        :return: a networkx graph of the graph
        """
        return nx.read_weighted_edgelist("graph.txt", nodetype=int, create_using=nx.DiGraph)

    def _graph_reference(self,window=2):
        """
        Transform a doc type into a graph and a word-to-node dictionary
        """
        numbering = 0
        denumbering = defaultdict()
        for sentence in self.doc:
            #print(sentence)
            for token in sentence:
                if token.dep_ != "ROOT":
                    if token.head.text not in self.reference:
                        self.reference[token.head.text] = numbering
                        denumbering[numbering] = token.head.text
                        numbering += 1
                    if token.text not in self.reference:
                        self.reference[token.text] = numbering
                        denumbering[numbering] = token.text
                        numbering += 1

                    '''if (self.reference[token.head.text], self.reference[token.text]) in self.graph:
                        self.graph[(self.reference[token.head.text], self.reference[token.text])] += 1
                    else:
                        self.graph[(self.reference[token.head.text], self.reference[token.text])] = 1'''
                    #self.graph[(self.reference[token.head.text], self.reference[token.text])] += 1

            for token in sentence:
                ct = token
                i = 0
                while(i<window):
                    self.graph[(self.reference[ct.head.text], self.reference[token.text])] += 1
                    ct = ct.head
                    i += 1
                    if ct.dep_ == "ROOT":
                        break

    def _graph_reference_bis(self, window=2):
        """
        Transform a doc type into a graph and a word-to-node dictionary
        """
        numbering = 0
        for i,sentence in enumerate(self.doc):
            G = nx.MultiDiGraph()
            localnumbering = { i: str(w) for i,w in enumerate(sentence) }
            for index,token in enumerate(sentence):
                if token.dep_ != "ROOT":
                    if token.head.text not in self.reference:
                        self.reference[token.head.text] = numbering
                        numbering += 1
                    if token.text not in self.reference:
                        self.reference[token.text] = numbering
                        numbering += 1

                if i==0:
                    G.add_edge(localdict[token.head.text],localdict[token.text])
                    #self.graph[(self.reference[token.head.text], self.reference[token.text])] += 1

            if i == 0:
                for e in list(nx.bfs_edges(G, source=localdict["cause"], depth_limit=window)):
                    print(self.reference[localnumbering[localdict["cause"]]],self.reference[localnumbering[e[1]]])
                    self.graph[(self.reference[localnumbering[localdict["cause"]]], self.reference[localnumbering[e[1]]])] += 1
                print(self.graph)

                #print(G.nodes,token.head.text,token.text)

            '''for source in G.nodes:
                for e in list(nx.bfs_edges(G, source=source,depth_limit=window)):
                    self.graph[(source, e[1])] += 1'''


    def _community_of_words(self, community, reference):
        """
        Uses the reference dictionary to return a dictionary of words to community with the community parameter.

        :return: a dictionary of community to words
        """
        correspondence = dict()
        for word, index in reference.items():
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
