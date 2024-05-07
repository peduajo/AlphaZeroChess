#pragma once

#include <iostream>
#include <torch/torch.h>
#include "game.h"
#include "params.h"
#include <utility>
#include <vector>
#include "mcts.h"

class MCTS;

class Node {
public: 
    ChessGameState game;
    AlphazeroParams args;
    Node* parent;
    torch::Tensor action_taken;
    float prior;
    int visit_count;
    int virtual_loss;
    float value_sum;
    bool is_expanded;
    int expanded_times = 0;
    std::unordered_map<std::string, Node*> children;//diccionario que tiene accion-hijo
    bool debug_node;
    std::mutex node_mutex;
    static std::atomic<int> contador_eliminaciones;
    static std::atomic<int> contador_creaciones;
    static std::vector<int> n_children;
public:
    Node(const ChessGameState& game, const AlphazeroParams& args, int visit_count, bool debug_node)
        : game(game),
          args(args),
          parent(nullptr),
          //action_taken(-1),
          prior(0.0),
          visit_count(visit_count),
          value_sum(0),
          virtual_loss(0),
          is_expanded(false),
          //children(std::vector<Node*>()),
          debug_node(debug_node){
            ++contador_creaciones;
          }
    
    Node(const ChessGameState& game, const AlphazeroParams& args, Node* parent, torch::Tensor action_taken, float prior)
        : game(game),
          args(args),
          parent(parent),
          action_taken(action_taken),
          prior(prior),
          visit_count(0),
          value_sum(0),
          virtual_loss(0),
          is_expanded(false),
          debug_node(false){
            ++contador_creaciones;
          }
    
    ~Node(){
        for (const auto& par: children){
            auto child = par.second;
            if(child != nullptr){
                delete child;
                child = nullptr;
                ++contador_eliminaciones;
            }
        }
    }

    //inline bool is_fully_expanded();
    Node* select();
    float get_ucb(Node* child);
    bool expand(const torch::Tensor& policy);
    void backpropagate(const float& value);
};