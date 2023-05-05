import spacy, scipy
import directedlouvain as dl
import networkx as nx
import networkit
import timeit
import pickle


class DirectedLouvain:
    graph = dict()
    reference = dict()
    doc = None
    louvain = None

    def __init__(self, text="text.txt", pipeline="en_core_web_sm", gamma=55):
        """
        Use the directed version of the louvain algorythme to analyse the text file.

        :param text: Path to the text file to analyse.
        :param pipeline: Specify the spacy pipeline to use.

        See also:
        `Spacy homepage <https://spacy.io/models>`_
        """
        nlp = spacy.load(pipeline)

        if isinstance(text, str):
            text_str = self._read_text(text)
        else:
            text_str = self._read_list(text)

        # make and write the graph inside the graph_text.txt file and generate a dictionary of words to nodes
        self.doc = nlp(text_str)
        self._graph_reference()
        self._write_graph()

        # calcul communities
        start = timeit.default_timer()
        self.louvain = dl.Community(self.graph, weighted=True, gamma=gamma)
        self.louvain.run(verbose=False)
        stop = timeit.default_timer()
        community = self._community_of_words(self.louvain.last_level(), self.reference)
        print("Average community size: " + str(len(self.reference) / len(community)))
        print('Time for community detection: ', stop - start)

        # save the matrix and dictionary inside the data.pk file
        self._save_data()

    def get_community(self):
        """
        Create a networkit graph community to feed the extract_embeddings function of the sinr library

        :return: a networkit type community graph
        """
        communities = self.louvain.last_level()
        partition = networkit.Partition(len(self._get_networkx_graph()))
        for node, community in communities.items():
            partition.addToSubset(community, node)
        return partition

    def load_data(self):
        """
        load the data.pk file to return a matrix of the graph and a dictionary(word to node)

        :return: the graph as a matrix and the dictionary
        """
        dico = []
        matrix = []
        with open("data.pk", "rb") as savefile:
            dico, matrix = pickle.load(savefile)
        return matrix, dico

    def _save_data(self):
        """
        save the graph as a matrix and the word dictionary in the data.pk file
        """
        filename = "data.pk"
        with open(filename, 'wb') as savefile:
            pickle.dump((self.reference, scipy.sparse._coo.coo_matrix(nx.to_scipy_sparse_array(self._get_networkx_graph()))),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return filename

    def _get_networkx_graph(self):
        """
        Create a graph under the networkx format

        :return: a networkx graph of the analyze graph
        """
        return nx.read_weighted_edgelist("graph_text.txt", nodetype=int, create_using=nx.DiGraph)

    def _graph_reference(self):
        """
        Transform a doc type into a graph and dictionary referencer
        """
        numbering = 0
        for token in self.doc:
            if token.dep_ != "ROOT":
                if token.head.text not in self.reference:
                    self.reference[token.head.text] = numbering
                    numbering += 1
                if token.text not in self.reference:
                    self.reference[token.text] = numbering
                    numbering += 1
                if (self.reference[token.head.text], self.reference[token.text]) in self.graph:
                    self.graph[(self.reference[token.head.text], self.reference[token.text])] += 1
                else:
                    self.graph[(self.reference[token.head.text], self.reference[token.text])] = 1

    def _community_of_words(self, community, reference):
        """
        Use the reference dictionary to return a dictionary of words to community with the community parameter.

        :return: a dictionary of community to words
        """
        correspondence = dict()
        for word, index in reference.items():
            comm = community[index]
            correspondence.setdefault(comm, []).append(word)
        return correspondence

    def _read_text(self, fileToRead):
        """
        Read a .txt file and return a string with his content

        :return: A string of the file
        """
        file = open(fileToRead, "r")
        text = file.read()
        text = text.lower()
        file.close()
        return text

    def _read_list(self, listToRead):
        """
        Read a .txt file and return a string with his content

        :return: A string of the file
        """
        str = ""
        for i in listToRead:
            for j in i:
                str = str + j + " "
        return str

    def _write_graph(self):
        """
        Export a graph as a .txt format file
        """
        file = open("graph_text.txt", "w")
        for head, tail in self.graph:
            file.write(str(head) + " " + str(tail) + " " + str(self.graph[(head, tail)]) + "\n")
        file.close()


def _parseArgs():
    """
    Parse arguments when this library is use as a main.

    :return: A list of args is return
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parseArgs()
    filename = args.f[0] if args.f else "text.txt"
    pipeline = args.p[0] if args.p else "en_core_web_sm"
    gamma    = args.g[0] if args.g else 55
    DirectedLouvain(text="text.txt", pipeline=pipeline, gamma=gamma)
