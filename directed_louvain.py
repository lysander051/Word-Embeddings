import spacy
import directedlouvain as dl
import networkx as nx
import timeit


def _graph_reference(doc, graph, reference):
    """
    Transform a doc type into a graph and word referencer

    :param doc: a parsed list of word into doc type
    :param graph: an empty dictionary to get back the graph
    :param reference: an empty dictionary to get back a word to int reference
    """
    numbering = 0
    for token in doc:
        if token.dep_ != "ROOT":
            if token.head.text not in reference:
                reference[token.head.text] = numbering
                numbering += 1
            if token.text not in reference:
                reference[token.text] = numbering
                numbering += 1
            if (reference[token.head.text], reference[token.text]) in graph:
                graph[(reference[token.head.text], reference[token.text])] += 1
            else:
                graph[(reference[token.head.text], reference[token.text])] = 1


def _community_of_words(community, reference):
    """
    Use the reference dictionary to return a dictionary of words to community with the community parameter.

    :param community: a dictionary of type int, int that associate words to community
    :param reference: a dictionary of word to int
    :return: a dictionary of community to words
    """
    correspondence = dict()
    for word, index in reference.items():
        comm = community[index]
        correspondence.setdefault(comm, []).append(word)
    return correspondence


def _read_text(fileToRead):
    """
    Read a .txt file and return a string with his content

    :param fileToRead: The path to the file to read
    :return: A string of the file
    """
    file = open(fileToRead, "r")
    text = file.read()
    text = text.lower()
    file.close()
    return text


def _write_graph(graph):
    """
    Export a graph as a .txt format file

    :param graph: graph to export
    """
    file = open("graph_text.txt", "w")
    for head, tail in graph:
        file.write(str(head) + " " + str(tail) + " " +  "\n")
    file.close()


def directed_louvain(filename="text.txt", pipeline="en_core_web_sm"):
    """
    Use the directed version of the louvain algorythme to analyse the text file.

    :param filename: Path to the text file to analyse.
    :param pipeline: Specify the spacy pipeline to use.

    See also:
    `Spacy homepage <https://spacy.io/models>`_
    """
    nlp = spacy.load(pipeline)
    text = _read_text(filename)
    doc = nlp(text)

    start = timeit.default_timer()
    graph = dict()
    reference = dict()

    _graph_reference(doc, graph, reference)
    _write_graph(graph)

    # 0.17 louvain et run avec une dl.map
    louvain = dl.Community(graph, weighted=True, gamma=55)
    community = _community_of_words(louvain.run(), reference)
    print(community)
    print("Community average size: " + str(len(reference)/len(community)))
    stop = timeit.default_timer()
    print('Time: ', stop - start)


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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parseArgs()
    filename = args.f[0] if args.f else "text.txt"
    pipeline = args.p[0] if args.p else "en_core_web_sm"
    directed_louvain(filename=filename, pipeline=pipeline)


