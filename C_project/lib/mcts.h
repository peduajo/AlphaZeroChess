#pragma once

#include <iostream>
#include <torch/torch.h>
#include "game.h"
#include "params.h"
#include <utility>
#include <vector>
#include "node.h"
#include "ThreadPool.h"

class Node;

class BufferEvalManager {
public:
    torch::jit::script::Module model;
    torch::Device device;
    std::mutex buffer_eval_mutex;
    std::condition_variable buffer_eval_cond;
    std::vector<torch::Tensor> buffer_eval_inputs;
    std::vector< std::promise< std::pair<torch::Tensor, float> > > buffer_eval_promises;
    bool terminate_thread_buffer_eval;
    int batch_size;
    torch::ScalarType dtype;
public:
    BufferEvalManager(const torch::Device& device, int batch_size) 
        : device(device), batch_size(batch_size), terminate_thread_buffer_eval(false){
    }

    void buffer_eval_controller();
    std::pair<torch::Tensor, torch::Tensor> batch_inference(std::vector<torch::jit::IValue> inputs);
    void set_model(std::string model_path);
};

class MCTS {
public:
    torch::Device device;
    AlphazeroParams args;
    Node* root;
    Node* initial_state_node;
    int num_searches_per_thread;
    int created_nodes;
    ThreadPool pool;
    std::shared_ptr<BufferEvalManager> buffer_eval_manager;
public:
    MCTS(const AlphazeroParams& args,const torch::Device& device, std::shared_ptr<BufferEvalManager> buffer_eval_manager) 
        : args(args), initial_state_node(nullptr), pool(args.num_threads_mcts), device(device), buffer_eval_manager(buffer_eval_manager){
        num_searches_per_thread = args.num_searches / args.num_threads_mcts;
        created_nodes = 0;
    }

    torch::Tensor search(bool debug);
    void reset_game(bool degug);
    void update_root(std::string action);
    void clean_tree();
    void make_thread_simulations(int search_id);
    std::tuple<Node*, float, bool> select_and_check_terminal(Node* node);
};
