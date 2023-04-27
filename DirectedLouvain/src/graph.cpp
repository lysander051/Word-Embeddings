#include <string>
#include <cstring>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>
#include "../include/graph.hpp"

const unsigned int MAP_LIMIT = 5000000;
static unsigned int build_map(std::map<std::tuple<int, int>,int> , vector<unsigned long>&, vector<vector<pair<unsigned int,double> > >&, vector<vector<pair<unsigned int,double> > >&, bool, bool, bool, bool);

Graph::Graph() {
    this->nodes         = 0;
    this->arcs          = 0;
    this->total_weight  = 0;
    this->weighted      = false;

    this->outcoming_arcs.resize(0);
    this->incoming_arcs.resize(0);
    this->outcoming_weights.resize(0);
    this->incoming_weights.resize(0);
    this->outdegrees.resize(0);
    this->indegrees.resize(0);
}

Graph::Graph(std::map<std::tuple<int, int>,int> graph, bool weighted, bool reproducibility, bool renumbering, bool verbose) {
    vector<vector<pair<unsigned int,double> > > LOUT;
    vector<vector<pair<unsigned int,double> > > LIN;

    this->weighted = weighted;

    this->correspondance.resize(0);
    this->nodes = build_map(graph, this->correspondance, LOUT, LIN, this->weighted, renumbering, reproducibility, verbose);

    init_attributes(*this, LOUT, LIN, verbose);
}

Graph::Graph(const Graph &g) {
    this->weighted          = g.weighted; 

    this->nodes             = g.nodes;
    this->arcs              = g.arcs;
    this->total_weight      = g.total_weight;

    this->outcoming_arcs    = g.outcoming_arcs;
    this->incoming_arcs     = g.incoming_arcs;
    this->outdegrees        = g.outdegrees;
    this->indegrees         = g.indegrees;
    this->outcoming_weights = g.outcoming_weights;
    this->incoming_weights  = g.incoming_weights;
    this->correspondance    = g.correspondance;
}

double Graph::count_selfloops(unsigned int node) {
    assert(node<this->nodes);
    size_t p = this->out_neighbors(node);
    for (unsigned int i=0 ; i < this->out_degree(node) ; ++i) {
        if (this->outcoming_arcs[p+i]==node) {
            if (this->weighted)
                return this->outcoming_weights[p+i];
            else 
                return 1.;
        }
    }

    return 0.;
}

double Graph::weighted_out_degree(unsigned int node) {
    assert(node<this->nodes);
    if (!this->weighted)
        return this->out_degree(node);
    else {
        size_t p = this->out_neighbors(node);
        double res = 0;
        for (unsigned int i=0 ; i < this->out_degree(node) ; ++i) 
            res += this->outcoming_weights[p+i];
        return res;
    }
}

double Graph::weighted_in_degree(unsigned int node) {
    assert(node<this->nodes);
    if (!this->weighted)
        return this->in_degree(node);
    else {
        size_t p = this->in_neighbors(node);
        double res = 0;
        for (unsigned int i=0 ; i < this->in_degree(node) ; ++i) 
            res += this->incoming_weights[p+i];
        return res;
    }
}

// Friend and static functions are defered to a different file for readability 
#include "graph_friend_static.cpp"
