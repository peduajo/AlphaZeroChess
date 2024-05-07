#include "alphazero.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <c10/cuda/CUDACachingAllocator.h>
#include <future>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include <cmath>
#include <sys/stat.h>
#include <filesystem> // Requiere C++17

#include <iostream>
#include <memory>
#include <unistd.h> // Para fork() y getpid()
#include <sys/wait.h> // Para wait()

namespace fs = std::filesystem;


std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> AlphaZero::_self_play(bool debug){
    //std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> memory;
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>> memory;
    MCTS mcts(args, device, buffer_eval_manager);
    int player = 1;
    mcts.reset_game(false);

    int hist_outcome;
    torch::Tensor hist_neutral_state;
    torch::Tensor hist_action_probs;
    torch::Tensor hist_q_values;
    int hist_player;
    //std::tuple<torch::Tensor, torch::Tensor, int> tuple_games;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> tuple_games;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tuple;
    float temperature = args.temperature;
    torch::Tensor valid_moves_tensor;
    std::string decoded_action;

    float q_value;

    ChessGameState game;

    while (true){
        //std::cout << "Aplicando MCTS" << std::endl;
        torch::Tensor action_probs = mcts.search(debug);

        q_value = static_cast<float>(mcts.root->value_sum)/mcts.root->visit_count;
        auto root_encoded_state = game.get_encoded_state();
        tuple_games = std::make_tuple(root_encoded_state, action_probs, torch::tensor({q_value}), player);
        memory.push_back(tuple_games);

        if (memory.size() > args.tau_zero_plays){
            temperature = 0.05;
        }

        torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/temperature);
        temp_action_probs /= torch::sum(temp_action_probs);

        //hay que hacer este lío para pasarlo por multinomial aplanandolo y luego volviendo a conocer el índice en el tensor 3D
        int action_idx = torch::multinomial(temp_action_probs, 1, false).item<int>();
        auto policy_zeros = torch::zeros(game.action_size_flat);
        policy_zeros.index_put_({action_idx}, 1);
        policy_zeros = policy_zeros.view(game.action_size);
        torch::Tensor action_idx_tensor = torch::nonzero(policy_zeros).squeeze(0);
        if(debug){
            decoded_action = game.decode_action(action_idx_tensor);
        }
        std::string action_idx_str = game.tensorToString(action_idx_tensor);
        mcts.update_root(action_idx_str);

        //std::cout << "Obteniendo nuevo estado" << std::endl;
        game.set_next_state(action_idx_tensor);
        game.set_player();

        //std::cout << "Obteniendo valor y si es terminal" << std::endl;
        valid_moves_tensor = game.get_valid_moves();
        auto terminated = game.get_value_and_terminated(valid_moves_tensor);
        float value = terminated.first;
        bool is_terminal = terminated.second;

        if (debug){
            //std::cout << "Tensor de probabilidades" << action_probs << std::endl;
            //std::cout << "Tensor de probabilidades temperature" << temp_action_probs << std::endl;
            //auto decoded_action = game.decode_action(action_idx_tensor);
            std::cout << game.board << std::endl;
            std::cout << "Acción tomada decoded: " << decoded_action << std::endl;
            std::cout << "Acción tomada: " << action_idx_tensor << std::endl;
            std::cout << "Terminal: " << std::to_string(is_terminal) << std::endl;
            std::cout << "Value: " << q_value << std::endl;
            std::cout << "----------------------------------" << std::endl;
        }

        if(q_value < args.v_surrender){
            value = -1;
            is_terminal = true;
            if(debug){
                std::cout << "Partida terminada por rendición!" << std::endl;
            }
        }


        if (is_terminal){
            //std::cout << "Devolviendo datos de partida terminada" << std::endl;
            std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> return_memory;

            //if (debug){
            //std::cout << state << std::endl;
            //std::cout << is_terminal << std::endl;
            //}
            
            for (const auto& m : memory) {
                //std::tie(hist_neutral_state, hist_action_probs, hist_player) = m;
                std::tie(hist_neutral_state, hist_action_probs, hist_q_values, hist_player) = m;
                if (hist_player == player){
                    hist_outcome = value;
                }else{
                    hist_outcome = game.get_opponent_value(value);
                }
                auto hist_outcome_tensor = torch::tensor({hist_outcome});
                //auto hist_encoded_state = game.get_encoded_state(hist_neutral_state, hist_player);
                tuple = std::make_tuple(hist_neutral_state, hist_action_probs, hist_q_values, hist_outcome_tensor);
                return_memory.push_back(tuple);
            }
            mcts.clean_tree();
            std::cout << "Nodos eliminados: " << mcts.initial_state_node->contador_eliminaciones << std::endl;
            std::cout << "Nodos creados: " << mcts.initial_state_node->contador_creaciones << std::endl;
            return return_memory;
        }

        player = (player == 1) ? 0 : 1;
    }
}


void AlphaZero::_save_memory(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> memory){
    //guardada de datos
    auto dir_path = base_path_data + std::to_string(iteration_idx);
    auto dir_path_states = dir_path + "/states/";
    auto dir_path_policy = dir_path + "/policy/";
    auto dir_path_q_values = dir_path + "/q_values/";
    auto dir_path_values = dir_path + "/values/";

    std::vector<torch::Tensor> state_vec;
    std::vector<torch::Tensor> policy_vec;
    std::vector<torch::Tensor> q_val_vec;
    std::vector<torch::Tensor> value_vec;

    torch::Tensor st;
    torch::Tensor p;
    torch::Tensor q;
    torch::Tensor v;

    for (const auto& m : memory) {
        std::tie(st, p, q, v) = m;
        state_vec.push_back(st);
        policy_vec.push_back(p);
        q_val_vec.push_back(q);
        value_vec.push_back(v);
    }

    auto dir_path_states_file = dir_path_states + std::to_string(process_idx) + ".zip";
    auto states_tensor = torch::stack(state_vec, 0);
    auto bytes_s = torch::jit::pickle_save(states_tensor);
    std::ofstream fout_s(dir_path_states_file, std::ios::out | std::ios::binary);
    fout_s.write(bytes_s.data(), bytes_s.size());
    fout_s.close();
    //std::cout << states_tensor.sizes() << std::endl;
    //torch::save(states_tensor, dir_path_states);

    auto dir_path_policy_file = dir_path_policy + std::to_string(process_idx) + ".zip";
    auto policy_tensor = torch::stack(policy_vec, 0);
    //torch::save(policy_tensor, dir_path_policy);
    auto bytes_p = torch::jit::pickle_save(policy_tensor);
    std::ofstream fout_p(dir_path_policy_file, std::ios::out | std::ios::binary);
    fout_p.write(bytes_p.data(), bytes_p.size());
    fout_p.close();

    auto dir_path_q_value_file = dir_path_q_values + std::to_string(process_idx) + ".zip";
    auto q_value_tensor = torch::stack(q_val_vec, 0);
    //torch::save(value_tensor, dir_path_value);
    auto bytes_v = torch::jit::pickle_save(q_value_tensor);
    std::ofstream fout_v(dir_path_q_value_file, std::ios::out | std::ios::binary);
    fout_v.write(bytes_v.data(), bytes_v.size());
    fout_v.close();

    auto dir_path_value_file = dir_path_values + std::to_string(process_idx) + ".zip";
    auto value_tensor = torch::stack(value_vec, 0);
    //torch::save(value_tensor, dir_path_value);
    auto bytes_q = torch::jit::pickle_save(value_tensor);
    std::ofstream fout_q(dir_path_value_file, std::ios::out | std::ios::binary);
    fout_q.write(bytes_q.data(), bytes_q.size());
    fout_q.close();
}


void AlphaZero::_save_memory_vm(const ChessGameState& game){
    //guardada de datos
    auto dir_path = base_path_data + "vm_data";
    auto dir_path_i = dir_path + "/inputs/";
    auto dir_path_o = dir_path + "/outputs/";


    std::vector<torch::Tensor> inputs;
    std::vector<torch::Tensor> outputs;
    torch::Tensor input;
    torch::Tensor output;

    for (const auto& m : game.positions_data) {
        std::tie(input, output) = m;
        inputs.push_back(input);
        outputs.push_back(output);
    }

    auto inputs_tensor = torch::stack(inputs, 0);
    auto outputs_tensor = torch::stack(outputs, 0);

    auto dir_path_inputs = dir_path_i + std::to_string(process_idx) + ".zip";
    std::cout << dir_path_inputs << std::endl;
    auto bytes_s = torch::jit::pickle_save(inputs_tensor);
    std::ofstream fout_s(dir_path_inputs, std::ios::out | std::ios::binary);
    fout_s.write(bytes_s.data(), bytes_s.size());
    fout_s.close();

    auto dir_path_outputs = dir_path_o + std::to_string(process_idx) + ".zip";
    std::cout << dir_path_outputs << std::endl;
    auto bytes_p = torch::jit::pickle_save(outputs_tensor);
    std::ofstream fout_p(dir_path_outputs, std::ios::out | std::ios::binary);
    fout_p.write(bytes_p.data(), bytes_p.size());
    fout_p.close();
}


void AlphaZero::_generate_self_play_data(std::string model_path){
    ChessGameState game; 

    std::cout << "Estamos en el proceso hijo:" << process_idx << std::endl;
    auto tiempo_inicio = std::chrono::high_resolution_clock::now();

    try {
        buffer_eval_manager->set_model(model_path);
    }catch (const c10::Error& e) {
        std::cerr << "Error al cargar el modelo: " << e.what() << std::endl;
        // Aquí puedes decidir si quieres hacer algo más antes de terminar, como limpiar recursos.
        std::exit(EXIT_FAILURE); // Termina el proceso con un código de error.
    }

    std::thread controller_thread([this] { this->buffer_eval_manager->buffer_eval_controller(); });

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> memory;

    std::vector<std::future<std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>>> futures;
    for (int i = 0; i < args.n_selfplay_iterations; ++i) {
        //int idx_buffer = i % args.num_multiprocessing;
        //auto buffer_selected = buffer_eval_list[idx_buffer];
        futures.emplace_back(
            pool_games.enqueue([this] { 
                return this->_self_play(false);
            })
        );         
    }

    for (auto& f : futures) {
        auto match_memory = f.get();  // Obtener el resultado de cada hilo
        memory.insert(memory.end(), match_memory.begin(), match_memory.end());
        std::cout << "Tamaño de la memoria: " << memory.size() << std::endl;
    }

    auto tiempo_fin = std::chrono::high_resolution_clock::now();
    auto duracion = std::chrono::duration_cast<std::chrono::microseconds>(tiempo_fin - tiempo_inicio);

    // Convierte la duración a segundos
    double segundos = static_cast<double>(duracion.count()) / 1'000'000.0;
    double minutos = segundos / 60.0;

    float steps_avg = memory.size()/args.n_selfplay_iterations;

    std::cout << "Timestamp: " << minutos << " | Partidas jugadas: " << args.n_selfplay_iterations << "| juegos medios por partida: " << steps_avg << std::endl;

    this->_save_memory(memory);
    //this->_save_memory_vm(game);

    buffer_eval_manager->terminate_thread_buffer_eval = true;
    if (controller_thread.joinable()) {
        controller_thread.join(); // Espera a que el hilo controlador termine
    }
}


void AlphaZero::pipeline_self_play(){

    std::string model_path = "../models/model_" + std::to_string(iteration_idx) + ".ts";

    auto dir_path = base_path_data + std::to_string(iteration_idx);

    // Comprobar si el directorio existe
    if (process_idx == 0){
        struct stat info;
        if (stat(dir_path.c_str(), &info) != -1) {
            // Si el directorio existe, eliminarlo
            try {
                fs::remove_all(dir_path); // Esta función elimina el directorio y su contenido
            } catch (const fs::filesystem_error& e) {
                throw std::runtime_error("Error al eliminar el directorio");
            }
        }

        auto dir_path_states = dir_path + "/states";
        auto dir_path_policy = dir_path + "/policy";
        auto dir_path_q_values = dir_path + "/q_values";
        auto dir_path_values = dir_path + "/values";
        auto dir_path_clean = dir_path + "/clean";

        const char* cDirPath = dir_path.c_str();
        const char* cDirPathStates = dir_path_states.c_str();
        const char* cDirPathPolicy = dir_path_policy.c_str();
        const char* cDirPathValues = dir_path_values.c_str();
        const char* cDirPathQValues = dir_path_q_values.c_str();
        const char* cDirPathClean = dir_path_clean.c_str();

        if (mkdir(cDirPath, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathStates, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathPolicy, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathQValues, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathValues, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathClean, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
    }
    
    this->_generate_self_play_data(model_path);
}
