#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <utility>
#include "mcts.h"
#include <memory>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <boost/archive/binary_oarchive.hpp>
#include <future>
#include "node.h"
#include <fstream>

constexpr auto max_wait_time_inference = std::chrono::milliseconds(10); // Ejemplo: 100 ms

void MCTS::clean_tree(){
    //std::cout << "Procediendo a eliminar arbol" << std::endl;
    initial_state_node->contador_eliminaciones++;
    if (initial_state_node != nullptr){
        delete initial_state_node;
        initial_state_node = nullptr;
    }
}

void MCTS::reset_game(bool debug){
    ChessGameState game;

    initial_state_node = new Node(game, args, 1, debug);
    root = initial_state_node;
}

void MCTS::update_root(std::string action){
    auto it = root->children.begin();
    while (it != root->children.end()) {
        if (action != it->first) {
            it->second->contador_eliminaciones++;
            delete it->second; // Liberar la memoria del nodo
            it = root->children.erase(it); // Elimina el elemento del mapa y actualiza el iterador al siguiente elemento
        } else {
            ++it; // Solo avanzar el iterador si no se elimina el elemento actual
        }
    }

    root = root->children[action];
}


torch::Tensor MCTS::search(bool debug){
    torch::Tensor obs_state;
    torch::Tensor policy;
    torch::Tensor valid_moves;
    float value;
    bool is_terminal;
    std::pair<bool, float> terminated;

    obs_state = root->game.get_encoded_state();

    std::promise< std::pair<torch::Tensor, float> > promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> guard(buffer_eval_manager->buffer_eval_mutex);
        buffer_eval_manager->buffer_eval_inputs.push_back(std::move(obs_state));
        buffer_eval_manager->buffer_eval_promises.push_back(std::move(promise));
        if (buffer_eval_manager->buffer_eval_inputs.size() >= buffer_eval_manager->batch_size) {
            buffer_eval_manager->buffer_eval_cond.notify_one();
        }
    }

    auto prediction = future.get();
    policy = prediction.first.view(root->game.action_size);

    auto alpha_dirichlet_tensor = torch::ones(root->game.action_size, torch::dtype(torch::kFloat32)) * args.dirichlet_alpha;

    auto dirichlet_noise = torch::_sample_dirichlet(alpha_dirichlet_tensor);

    auto base_policy = (1 - args.dirichlet_epsilon) * policy;
    auto noise_policy = args.dirichlet_epsilon * dirichlet_noise;

    policy = base_policy + noise_policy;

    if(root->game.valid_moves_cache.defined()){
        valid_moves = root->game.valid_moves_cache;
    }else{
        valid_moves = root->game.get_valid_moves();
        root->game.valid_moves_cache = valid_moves;
    }

    policy *= valid_moves;
    policy /= torch::sum(policy);

    if (!root->is_expanded){
        root->expand(policy);
    }else{
        auto non_zero_indices = torch::nonzero(policy);
        for(int i=0; i<non_zero_indices.sizes()[0]; ++i){
            auto action_idx = non_zero_indices.index({i, "..."});
            std::string key_action = root->game.tensorToString(action_idx);
            int idx_dim_0 = action_idx.index({0}).item<int>();
            int idx_dim_1 = action_idx.index({1}).item<int>();
            int idx_dim_2 = action_idx.index({2}).item<int>();
            float prob = policy.index({idx_dim_0, idx_dim_1, idx_dim_2}).item<float>();
            root->children[key_action]->prior = prob;
        }
    }
    std::vector<std::future<void>> futures;

    for (int i = 0; i < args.num_searches; ++i) {
        futures.emplace_back(
            pool.enqueue([this, i] { 
                this->make_thread_simulations(i);
            })
        );
    }

    for (auto & future : futures) {
        future.get(); // Espera a que el hilo asociado termine
    }

    torch::Tensor action_probs = torch::zeros(root->game.action_size);
    for (const auto& par: root->children){
        auto child = par.second;
        int idx_dim_0 = child->action_taken.index({0}).item<int>();
        int idx_dim_1 = child->action_taken.index({1}).item<int>();
        int idx_dim_2 = child->action_taken.index({2}).item<int>();
        action_probs.index_put_({idx_dim_0, idx_dim_1, idx_dim_2}, child->visit_count);
    }
    action_probs /= torch::sum(action_probs);

    return action_probs.view(-1); 
}

std::tuple<Node*, float, bool> MCTS::select_and_check_terminal(Node* node){
    while (node->is_expanded){
        node = node->select();
    }
    torch::Tensor valid_moves;
    
    {
        std::lock_guard<std::mutex> lock(node->node_mutex);
        if(node->game.valid_moves_cache.defined()){
            valid_moves = node->game.valid_moves_cache;
        }else{
            valid_moves = node->game.get_valid_moves();
            node->game.valid_moves_cache = valid_moves;
        }
    }
    
    auto terminated = node->game.get_value_and_terminated(valid_moves);

    float value = terminated.first;
    bool is_terminal = terminated.second;

    value = node->game.get_opponent_value(value);

    return {node, value, is_terminal};
}


void MCTS::make_thread_simulations(int search_id){
    torch::Tensor obs_state;
    torch::Tensor policy;
    torch::Tensor valid_moves;

    Node* node = root;
    auto selection_data = this->select_and_check_terminal(node);
    node = std::get<0>(selection_data);
    auto value = std::get<1>(selection_data);
    auto is_terminal = std::get<2>(selection_data);

    bool thread_expanded = false;

    while (!is_terminal && !thread_expanded){
        obs_state = node->game.get_encoded_state();

        std::promise< std::pair<torch::Tensor, float> > promise;
        auto future = promise.get_future();

        {
            std::lock_guard<std::mutex> guard(buffer_eval_manager->buffer_eval_mutex);
            buffer_eval_manager->buffer_eval_inputs.push_back(std::move(obs_state));
            buffer_eval_manager->buffer_eval_promises.push_back(std::move(promise));
            if (buffer_eval_manager->buffer_eval_inputs.size() >= buffer_eval_manager->batch_size) {
                buffer_eval_manager->buffer_eval_cond.notify_one();
            }
        }

        auto prediction = future.get();

        policy = prediction.first.view(node->game.action_size);
        value = prediction.second;

        {
            std::lock_guard<std::mutex> lock(node->node_mutex);
            if(node->game.valid_moves_cache.defined()){
                valid_moves = node->game.valid_moves_cache;
            }else{
                valid_moves = node->game.get_valid_moves();
                node->game.valid_moves_cache = valid_moves;
            }
        }

        policy *= valid_moves;
        policy /= torch::sum(policy);

        thread_expanded = node->expand(policy);
        if (!thread_expanded){
            auto selection_data = this->select_and_check_terminal(node);
            node = std::get<0>(selection_data);
            value = std::get<1>(selection_data);
            is_terminal = std::get<2>(selection_data);
        }
    }

    node->backpropagate(value);
}


void BufferEvalManager::set_model(std::string model_path){
    std::cout << "Cargando modelo del path: " << model_path << std::endl;
    model = torch::jit::load(model_path);
    std::cout << "Modelo cargado!" << std::endl;
    model.to(device);
    model.eval();
}


std::pair<torch::Tensor, torch::Tensor> BufferEvalManager::batch_inference(std::vector<torch::jit::IValue> inputs){
    torch::NoGradGuard no_grad;

    auto inference = model.forward(inputs).toTuple();
    auto inference_elements = inference->elements();
    auto new_policies = inference_elements[0].toTensor();
    new_policies = torch::softmax(new_policies, 1).to(torch::kCPU);
    auto new_values = inference_elements[1].toTensor().to(torch::kCPU);


    return std::make_pair(new_policies, new_values);
}

void BufferEvalManager::buffer_eval_controller(){
    std::unique_lock<std::mutex> lock(buffer_eval_mutex);
    std::vector<torch::jit::IValue> inputs;
    while(true){
        while (!terminate_thread_buffer_eval && buffer_eval_inputs.size() < batch_size) {
            if (buffer_eval_cond.wait_for(lock, max_wait_time_inference, [this]{ return buffer_eval_inputs.size() >= batch_size || terminate_thread_buffer_eval; })) {
                break;
            } else {
                if (!buffer_eval_inputs.empty()) {
                    break;
                }
            }
        }
        if (terminate_thread_buffer_eval && buffer_eval_inputs.empty()) {
            break;
        }
        auto batch_encoded_states = torch::concat(buffer_eval_inputs, 0).to(device);
        inputs.push_back(batch_encoded_states);

        auto predictions = this->batch_inference(inputs);
        auto policies = predictions.first;
        auto values = predictions.second;
        for (int i = 0; i < buffer_eval_inputs.size(); ++i) {
            auto policy = policies.index({i, "..."});
            auto value = values[i].item<float>();
            auto pair_pred = std::make_pair(policy, value);
            buffer_eval_promises[i].set_value(pair_pred);
        }

        buffer_eval_inputs.clear();
        buffer_eval_promises.clear();
        inputs.clear();
    }
}