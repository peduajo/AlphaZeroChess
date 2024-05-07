#include <torch/torch.h>
#include <iostream>
#include "lib/params.h"
#include "lib/alphazero.h"
#include <chrono>
#include <torch_tensorrt/logging.h>

int main(int argc, const char* argv[]) {
    AlphazeroParams alphazero_params;
    alphazero_params.n_selfplay_iterations = 4;
    alphazero_params.num_searches = 400;
    alphazero_params.batch_size = 128;
    alphazero_params.temperature = 1.0;
    alphazero_params.dirichlet_epsilon = 0.25;
    alphazero_params.dirichlet_alpha = 0.3;
    alphazero_params.c_puct = 2.0;
    alphazero_params.num_threads_mcts = 8;
    alphazero_params.num_threads_games = 15;
    alphazero_params.tau_zero_plays = 80;
    alphazero_params.v_surrender = -0.75;

    if (argc != 3) {
        std::cerr << "usage: alphazero_train <iteration_idx> <process_idx>\n";
        return -1;
    }

    int iteration_idx = std::stoi(argv[1]);
    int process_idx = std::stoi(argv[2]);

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kWARNING);

    AlphaZero alphazero(alphazero_params, iteration_idx, process_idx, device);
    alphazero.pipeline_self_play();

    return 0;
}