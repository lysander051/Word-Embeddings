#include "../include/community.hpp"
#include <map>
#include <climits>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

// Static function renumbering communities from 0 to k-1 (returns k)
static unsigned int renumber_communities(const Community &c, vector< int > &renumber);
static void update_levels(const Community &c, vector< vector<int> > &levels, int level);

Community::Community(std::map<std::pair<int,int>, int> &graph, bool weighted, const double precision, const double gamma, bool reproducibility, bool renumbering, bool sorted) {
    this->g                 = new Graph(graph, weighted, reproducibility, renumbering);
    this->precision         = precision;
    this->gamma             = gamma;
    this->sorted            = sorted;

    init_attributes();
}

void Community::init_attributes() {
    this->size              = g->nodes;
    this->node_to_community.resize(this->size); 
    this->communities_arcs.resize(this->size);

    for (unsigned int i = 0; i < this->size; ++i) {
        // i belongs to its own community
        this->node_to_community[i]                      = i;
        // the total number of edges inside the community corresponds to 
        // the number of self-loops of node i (after at least one step)
        this->communities_arcs[i].total_arcs_inside     = g->count_selfloops(i);
        this->communities_arcs[i].total_outcoming_arcs  = g->weighted_out_degree(i);
        this->communities_arcs[i].total_incoming_arcs   = g->weighted_in_degree(i);
    }
}

Community::~Community() {
    delete this->g;
    delete this->community_graph;
}

double Community::modularity() {
    double q = 0.;
    double m = g->get_total_weight();
    for (unsigned int i = 0; i < size; ++i) {
        if (this->communities_arcs[i].total_incoming_arcs > 0 || this->communities_arcs[i].total_outcoming_arcs > 0) {
            double total_outcoming_arcs_var     = this->communities_arcs[i].total_outcoming_arcs / m;
            double total_incoming_arcs_var      = this->communities_arcs[i].total_incoming_arcs / m;
            q                                   += this->communities_arcs[i].total_arcs_inside / m - this->gamma * (total_outcoming_arcs_var * total_incoming_arcs_var);
        }
    }
    return q;
}

void Community::display_partition() {
    vector < int > renumber(size, -1);
    renumber_communities(*this, renumber);
    // Marking the beginning of new level
    cout << -1 << " " << -1 << endl;
    for (unsigned int i = 0; i < size; ++i)
        cout << (this->community_graph)->correspondance[i] << " " << renumber[this->node_to_community[i]] << endl;
}

void Community::partition_to_graph() {
    // Renumbering communities
    vector<int> renumber(size, -1);
    unsigned int f = renumber_communities(*this, renumber);

    // Computing communities (k lists of nodes)
    vector <vector<int>> comm_nodes(f);
    for (unsigned int node = 0; node < size; ++node) 
        comm_nodes[renumber[this->node_to_community[node]]].push_back(node);

    // Computing contracted weighted graph
    Graph *g2 = new Graph();
    g2->nodes = comm_nodes.size();
    g2->weighted = true;

    // Correspondance is set to identity since the graph is renumbered from 0 to k-1 (communities)
    for(unsigned int i = 0; i < g2->nodes; ++i)
        g2->correspondance.push_back(i);

    g2->outdegrees.resize(g2->nodes);
    g2->indegrees.resize(g2->nodes);

    unsigned int out_neighbor, out_neighboring_community;
    double out_weight;
    unsigned int in_neighbor, in_neighboring_community;
    double in_weight;

    // Computing arcs between communities
    for (size_t comm = 0; comm < g2->nodes; ++comm) {
        map <int,double> m_out, m_in;
        size_t comm_size = comm_nodes[comm].size();
        for (unsigned int node = 0; node < comm_size; ++node) {
            // Out-neighbors
            size_t p = (this->community_graph)->out_neighbors(comm_nodes[comm][node]);
            unsigned int deg = (this->community_graph)->out_degree(comm_nodes[comm][node]);
            // Looking for communities of every out-neighbor of node and then storing/updating weighted out-degrees
            for (unsigned int i = 0; i < deg; ++i) {
                out_neighbor = (this->community_graph)->outcoming_arcs[p + i];
                out_neighboring_community = renumber[this->node_to_community[out_neighbor]];
                out_weight = ((this->community_graph)->weighted) ? (this->community_graph)->outcoming_weights[p + i] : 1.f;

                auto it_out = m_out.find(out_neighboring_community);
                if (it_out == m_out.end())
                    m_out.insert(make_pair(out_neighboring_community, out_weight));
                else
                    it_out -> second += out_weight;
            }

            // In-neighbors
            size_t p_in = (this->community_graph)->in_neighbors(comm_nodes[comm][node]);
            deg = (this->community_graph)->in_degree(comm_nodes[comm][node]);
            // Looking for communities of every in-neighbor of node and then storing/updating weighted in-degrees
            for (unsigned int i = 0; i < deg; ++i) {
                in_neighbor = (this->community_graph)->incoming_arcs[p_in + i];
                in_neighboring_community = renumber[this->node_to_community[in_neighbor]];
                in_weight = ((this->community_graph)->weighted) ? (this->community_graph)->incoming_weights[p_in + i] : 1.f;

                auto it_in = m_in.find(in_neighboring_community);
                if (it_in == m_in.end())
                    m_in.insert(make_pair(in_neighboring_community, in_weight));
                else
                    it_in -> second += in_weight;
            }
        }

        // Building outcoming and incoming arcs according to previously computed weights
        g2->outdegrees[comm] = (comm == 0) ? m_out.size() : g2->outdegrees[comm - 1] + m_out.size();
        g2->arcs += m_out.size();

        for (auto it_out = m_out.begin(); it_out != m_out.end(); ++it_out) {
            g2->total_weight += it_out -> second;
            g2->outcoming_arcs.push_back(it_out -> first);
            g2->outcoming_weights.push_back(it_out -> second);
        }

        g2->indegrees[comm] = (comm == 0) ? m_in.size() : g2->indegrees[comm - 1] + m_in.size();

        for (auto it_in = m_in.begin(); it_in != m_in.end(); ++it_in) {
            g2->incoming_arcs.push_back(it_in -> first);
            g2->incoming_weights.push_back(it_in -> second);
        }
    }

    // Updating graph attribute with computed graph g2
    delete this->community_graph;
    this->community_graph = g2;

    // Updating other attributes according to constructed graph g
    this->size           = this->community_graph->nodes;
    this->node_to_community.resize(this->size); 

    for (unsigned int i = 0; i < this->size; ++i) {
        this->node_to_community[i]                      = i; 
        this->communities_arcs[i].total_arcs_inside     = this->community_graph->count_selfloops(i);
        this->communities_arcs[i].total_outcoming_arcs  = this->community_graph->weighted_out_degree(i);
        this->communities_arcs[i].total_incoming_arcs   = this->community_graph->weighted_in_degree(i);
    }
}

bool Community::one_level(double &modularity) {
    int nb_moves = 0;
    bool improvement = false;
    double current_modularity = this->modularity();
    double delta;

    // Order in which to proceed nodes of the graph...
    vector < int > random_order(size);
    for (unsigned int i = 0; i < size; ++i)
        random_order[i] = i;

    // ... randomized: (Directed) Louvain's algorithm is not deterministic
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    if(sorted)
        shuffle(random_order.begin(), random_order.end(), std::default_random_engine(seed));

    // Vectors containing weights and positions of neighboring communities
    vector<double> neighbor_weight(this->size,-1);
    vector<unsigned int> positions_neighboring_communities(size);
    // Every node neighbors its own community
    unsigned int neighboring_communities = 1;

    do {
        nb_moves = 0;
        delta = 0.;
        // For each node: remove it from its community and insert it in the best community (if any)
        for (unsigned int node_tmp = 0; node_tmp < size; ++node_tmp) {
            int node = random_order[node_tmp];
            int node_community = this->node_to_community[node];
            double weighted_out_degree  = (this->community_graph)->weighted_out_degree(node);
            double weighted_in_degree   = (this->community_graph)->weighted_in_degree(node);
            double self_loops           = (this->community_graph)->count_selfloops(node);

            // Computating all neighboring communities of current node (the number of such communities is stored in neighboring_communities)
            list_neighboring_communities(node, *this, neighbor_weight, positions_neighboring_communities, neighboring_communities);

            // Gain from removing node from its current community
            double removal = gain_from_removal(*this, node, node_community, neighbor_weight[node_community], weighted_out_degree, weighted_in_degree);
            remove(*this, node, node_community, neighbor_weight[node_community]+self_loops, weighted_out_degree, weighted_in_degree);

            // Default choice for future insertion is the former community
            int best_community = node_community;
            double best_nbarcs = neighbor_weight[node_community];
            double best_increase = 0.;

            // Computing modularity gain for all neighboring communities
            for (unsigned int i = 0; i < neighboring_communities; ++i) {
                // (Gain from) inserting note to neighboring community
                double insertion = gain_from_insertion(*this, node, positions_neighboring_communities[i], neighbor_weight[positions_neighboring_communities[i]], weighted_out_degree, weighted_in_degree);
                double increase = insertion;
                if(increase > best_increase) {
                    best_community = positions_neighboring_communities[i];
                    best_nbarcs = neighbor_weight[best_community];
                    best_increase = increase;
                }
            }
            // Inserting node in the nearest community
            insert(*this, node, best_community, best_nbarcs+self_loops, weighted_out_degree, weighted_in_degree);

            // If a move was made then we do one more step
            if (best_community != node_community) {
                delta+=best_increase+removal;
                improvement = true;
                ++nb_moves;
            }
        }
        
        // Computing the difference between the two modularities
        current_modularity = delta + current_modularity;
        // Printing modularity function and modularity increases

    } while (nb_moves > 0 && delta > precision);

    modularity = current_modularity;
    return improvement;
}

map<int, int> Community::get_level(int level){
    assert(level >= 0 && level < (int)this->levels.size());
    vector < int > n2c(this->g->nodes);
    map<int, int> lvl;

    for (unsigned int i = 0; i < this->g->nodes; i++)
        n2c[i] = i;

    for (int l = 0; l < level; l++)
        for (unsigned int node = 0; node < this->g->nodes; node++)
            n2c[node] = this->levels[l][n2c[node]];

    for (unsigned int node = 0; node < this->g->nodes; node++)
        lvl[(this->g)->correspondance[node]] = n2c[node];
    return lvl;
}

void Community::print_level(int level) {
    assert(level >= 0 && level < (int)this->levels.size());
    vector < int > n2c(this->g->nodes);

    for (unsigned int i = 0; i < this->g->nodes; i++)
        n2c[i] = i;

    for (int l = 0; l < level; l++)
        for (unsigned int node = 0; node < this->g->nodes; node++)
            n2c[node] = this->levels[l][n2c[node]];

    for (unsigned int node = 0; node < this->g->nodes; node++) 
        cout << (this->g)->correspondance[node] << " " << n2c[node] << endl;
}

map<int,int> Community::run(bool verbose, const int& display_level) {
    int level = 0;
    double mod = this->modularity();
    vector < int > corres(0);

    bool improvement = true;
    this->community_graph   = new Graph(*(this->g));
    this->init_attributes();
    do {
        if (verbose) {
            cerr << "level " << level << ":\n";
            cerr << "  network size: " <<
                this->community_graph->get_nodes() << " nodes, " <<
                this->community_graph->get_arcs() << " arcs, " <<
                this->community_graph->get_total_weight() << " weight." << endl;
        }

        // Directed Louvain: main procedure
        double new_mod = 0;
        improvement = this->one_level(new_mod);
        // Maintaining levels
        levels.resize(++level);
        update_levels(*this, levels, level-1);
        if ((level == display_level || display_level == -1) && verbose)
            this->display_partition();
        // Updating the graph to computer hierarchical structure
        this->partition_to_graph();
        if (verbose)
            cerr << "  modularity increased from " << mod << " to " << new_mod << endl;

        mod = new_mod;
    } while (improvement);
    if (display_level == -2 && verbose)
        print_level(levels.size()-1);
    return get_level(levels.size()-1);
}

// Friend and static functions are defered to a different file for readability 
#include "community_friend_static.cpp"
