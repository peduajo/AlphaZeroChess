#include "mcts.h"
#include <memory>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include "node.h"

std::atomic<int> Node::contador_eliminaciones = 0;
std::atomic<int> Node::contador_creaciones = 0;
std::vector<int> Node::n_children;

bool Node::expand(const torch::Tensor& policy){
    std::lock_guard<std::mutex> lock(node_mutex);
    if (!is_expanded){

        auto non_zero_indices = torch::nonzero(policy);
        for(int i=0; i<non_zero_indices.sizes()[0]; ++i){
            auto action_idx = non_zero_indices.index({i, "..."});
            int idx_dim_0 = action_idx.index({0}).item<int>();
            int idx_dim_1 = action_idx.index({1}).item<int>();
            int idx_dim_2 = action_idx.index({2}).item<int>();
            float prob = policy.index({idx_dim_0, idx_dim_1, idx_dim_2}).item<float>();
            std::string key_action = game.tensorToString(action_idx);

            ChessGameState game_child(game);
            game_child.set_next_state(action_idx);
            game_child.set_player();

            Node* child = new Node(game_child, args, this, action_idx, prob);
            this->children[key_action] = child;

        }
        is_expanded = true;
        expanded_times += 1;

        return true;
    }else{
        return false;
    }
}

Node* Node::select(){
    Node* best_child = nullptr;
    float best_ucb = -99999.0;

    for (const auto& par: children){
        auto child = par.second;
        float ucb = this->get_ucb(child);

        if (ucb > best_ucb){
            best_child = child;
            best_ucb = ucb;
        }
    }

    best_child->virtual_loss += 1;
    return best_child;
}

float Node::get_ucb(Node* child){
    float q_value = 0.0;
    if (child->visit_count > 0){
        float value_sum_adjusted = child->value_sum + child->virtual_loss;
        float q_value_prob = ((value_sum_adjusted / child->visit_count) + 1) / 2;

        q_value = 1 - q_value_prob;
    }
    float ucb = q_value + args.c_puct * (std::sqrt(visit_count) / (child->visit_count + 1)) * child->prior;
    return ucb;
}

void Node::backpropagate(const float& value){
    value_sum += value;
    visit_count += 1; 
    virtual_loss -= 1;

    float opponent_value = game.get_opponent_value(value);
    if (parent != nullptr){
        parent->backpropagate(opponent_value);
    }
}