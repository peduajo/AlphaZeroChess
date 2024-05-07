#include <torch/torch.h>
#include "lib/game.h"
#include <iostream>
#include "lib/params.h"
#include <chrono>
#include "lib/mcts.h"
#include <torch/script.h>

int test_double_check_mate_white(){
    ChessGameState game;
    game.cod_fem = "3rkb2/5p2/8/8/8/8/4B1K1/4R3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({6, 4, 23});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_double_check_mate_black(){
    ChessGameState game;
    game.cod_fem = "4k2r/5p1b/8/8/8/8/8/6NK b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({1, 7, 21});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_double_check_defense_white(){
    ChessGameState game;
    game.cod_fem = "3rk3/5pb1/8/8/8/8/4B1K1/4R3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({6, 4, 23});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;
    int expected_move = valid_moves_tensor.index({0, 4, 2}).item<int>();
    int sum_moves = torch::sum(valid_moves_tensor).item<int>();

    if (pair.first == 0 && expected_move == 1 && sum_moves == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_double_check_defense_black(){
    ChessGameState game;
    game.cod_fem = "1r2k3/1b3p2/8/8/8/8/P7/1KN5 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({1, 1, 19});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;
    int expected_move = valid_moves_tensor.index({7, 1, 6}).item<int>();
    int sum_moves = torch::sum(valid_moves_tensor).item<int>();

    if (pair.first == 0 && expected_move == 1 && sum_moves == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_drowning_pawns_white(){
    ChessGameState game;
    game.cod_fem = "7r/8/8/4k3/5p2/3b4/5P2/6K1 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({4, 5, 4});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_drowning_pawns_black(){
    ChessGameState game;
    game.cod_fem = "6k1/5p2/8/2B2P2/6K1/8/8/7R w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({3, 5, 0});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_drowning_stuck_pieces_white(){
    ChessGameState game;
    game.cod_fem = "7k/6pn/8/2B2P2/2B2K2/8/8/7R w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({3, 2, 3});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    std::cout << torch::nonzero(valid_moves_tensor) << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_drowning_simple_black(){
    ChessGameState game;
    game.cod_fem = "8/8/8/8/8/3k4/4p3/4K3 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({5, 3, 2});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_knight_only_check_mate_white(){
    ChessGameState game;
    game.cod_fem = "r1bqkb1r/pp1npppp/2p2n2/8/3PN3/8/PPP1QPPP/R1B1KBNR w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({4, 4, 56});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    std::cout << torch::nonzero(valid_moves_tensor) << std::endl;
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 1 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_knight_only_check_mate_black(){
    ChessGameState game;
    game.cod_fem = "8/2k5/8/8/3n4/8/PP6/KR6 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({4, 3, 61});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 1 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_knight_only_defense_check_mate_black(){
    ChessGameState game;
    game.cod_fem = "8/2k5/8/8/3n4/8/PP5R/KR6 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({4, 3, 61});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_knight_only_defense_check_mate_white(){
    ChessGameState game;
    game.cod_fem = "6rk/6pp/8/4K3/5N2/8/8/8 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({4, 5, 57});
    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_large_castling_doing_check(){
    ChessGameState game;
    game.cod_fem = "r2krq2/4p1pp/8/8/8/2N2N2/PPP2PP1/R3K3 w Qq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({7, 4, 14});
    int expected_move = valid_moves_tensor.index({7, 4, 14}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second and expected_move == 1 and game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_short_castling_white(){
    ChessGameState game;
    game.cod_fem = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({7, 4, 10});
    int expected_move = valid_moves_tensor.index({7, 4, 10}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second and expected_move == 1){
        return 1;
    }else{
        return 0;
    }
}


int test_long_castling_white(){
    ChessGameState game;
    game.cod_fem = "r2qkbnr/ppp2ppp/2n1p3/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({7, 4, 14});
    int expected_move = valid_moves_tensor.index({7, 4, 14}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second and expected_move == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_short_castling_black(){
    ChessGameState game;
    game.cod_fem = "rnbqk2r/pppp1ppp/5n2/2b1p3/2P5/2NP1N2/PP2PPPP/R1BQKB1R b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({0, 4, 10});
    int expected_move = valid_moves_tensor.index({0, 4, 10}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second and expected_move == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_long_castling_black(){
    ChessGameState game;
    game.cod_fem = "r3kbnr/pp1bpppp/2np4/qBp5/2P1P3/2N2N2/PP1P1PPP/R1BQ2KR b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();
    auto action_idx = torch::tensor({0, 4, 14});
    int expected_move = valid_moves_tensor.index({0, 4, 14}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and !pair.second and expected_move == 1){
        return 1;
    }else{
        return 0;
    }
}


int test_enPassant_white(){
    ChessGameState game;
    game.cod_fem = "rnbqkbnr/pp1ppppp/8/2p1P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    game.fentoBoard();
    std::cout << "current player: " << game.currentPlayer << std::endl;
    auto action_idx = torch::tensor({1, 3, 12});

    game.set_next_state(action_idx);
    game.set_player();
    std::cout << "current player: " << game.currentPlayer << std::endl;
    auto valid_moves_tensor = game.get_valid_moves();

    int expected_move = valid_moves_tensor.index({3, 4, 7}).item<int>();
    action_idx = torch::tensor({3, 4, 7});

    game.set_next_state(action_idx);
    game.set_player();
    std::cout << "current player: " << game.currentPlayer << std::endl;
    valid_moves_tensor = game.get_valid_moves();

    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    int target_piece = game.flatten_board.index({27}).item<int>();
    std::cout << game.board << std::endl;
    std::cout << expected_move << std::endl;

    if (pair.first == 0 && !pair.second && expected_move == 1 && target_piece == 0){
        return 1;
    }else{
        std::cout << "Fail!" << std::endl;
        return 0;
    }
}

int test_enPassant_black(){
    ChessGameState game;
    game.cod_fem = "rnbqkbnr/ppp1pppp/8/8/3p4/1P3N2/P1PPPPPP/RNBQKB1R w KQkq - 0 1";
    game.fentoBoard();
    auto action_idx = torch::tensor({6, 2, 8});

    game.set_next_state(action_idx);
    game.set_player();
    auto valid_moves_tensor = game.get_valid_moves();

    int expected_move = valid_moves_tensor.index({4, 3, 5}).item<int>();
    action_idx = torch::tensor({4, 3, 5});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();

    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    int target_piece = game.flatten_board.index({34}).item<int>();
    std::cout << game.board << std::endl;

    if (pair.first == 0 && !pair.second && expected_move == 1 && target_piece == 0){
        return 1;
    }else{
        std::cout << "Fail!" << std::endl;
        return 0;
    }
}

int test_cut_castling_short_white(){
    ChessGameState game;
    game.cod_fem = "r1bqkb1r/p1p1pppp/1p3n2/3p4/3P4/4PN2/PPP2PPP/RNBQK2R b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 2, 13});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    action_idx = torch::tensor({7, 4, 10});
    int expected_move = valid_moves_tensor.index({7, 4, 10}).item<int>();
    std::cout << game.target_squares_attacked_by_enemy << std::endl;

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_cut_castling_long_white(){
    ChessGameState game;
    game.cod_fem = "r1bqkb1r/pppppp1p/2n3pB/8/3P4/2NQ4/PPP1PPPP/R3KBNR b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 5, 11});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    std::cout << game.board << std::endl;

    action_idx = torch::tensor({7, 4, 14});
    int expected_move = valid_moves_tensor.index({7, 4, 14}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_cut_castling_short_black(){
    ChessGameState game;
    game.cod_fem = "rnbqk2r/pppp1ppp/5n2/4p3/4P3/bP6/P1PP1PPP/R1BQKBNR w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 2, 15});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    action_idx = torch::tensor({0, 4, 10});
    int expected_move = valid_moves_tensor.index({0, 4, 10}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}


int test_cut_castling_long_black(){
    ChessGameState game;
    game.cod_fem = "r3kbnr/ppp1pppp/2nq4/3p4/3P4/2N3Pb/PPP1PP1P/R1BQKB1R w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 5, 9});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    action_idx = torch::tensor({0, 4, 10});
    int expected_move = valid_moves_tensor.index({0, 4, 14}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_cut_castling_by_check(){
    ChessGameState game;
    game.cod_fem = "r1bq1rk1/ppp2ppp/2np1n2/2b5/2B5/2NP1N2/PPP2PPP/R1BQK2R b KQhq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 5, 6});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    action_idx = torch::tensor({0, 4, 10});
    int expected_move = valid_moves_tensor.index({7, 4, 10}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_avoided_by_moving_king_white(){
    ChessGameState game;
    game.cod_fem = "r1bq1rk1/ppp2ppp/2np1n2/2b5/2B5/2NP1N2/PPP2PPP/R1BQK2R w KQhq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 4, 7});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 2, 3});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 3, 3});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 3, 7});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    int expected_move = valid_moves_tensor.index({7, 4, 10}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_avoided_by_moving_king_black(){
    ChessGameState game;
    game.cod_fem = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    int expected_move = valid_moves_tensor.index({0, 4, 10}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_stuck_piece_white(){
    ChessGameState game;
    game.cod_fem = "rnbqkb1r/pppp1ppp/5n2/4p3/8/2NP4/PPP1PPPP/R1BQKBNR b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 5, 29});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();

    action_idx = torch::tensor({5, 2, 56});
    int expected_move = valid_moves_tensor.index({5, 2, 56}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_stuck_piece_black(){
    ChessGameState game;
    game.cod_fem = "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 5, 31});

    game.set_next_state(action_idx);
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();

    action_idx = torch::tensor({2, 2, 56});
    int expected_move = valid_moves_tensor.index({2, 2, 56}).item<int>();

    game.set_next_state(action_idx);
    game.set_player();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_promotion_queen_without_eating_white(){
    ChessGameState game;
    game.cod_fem = "8/2P5/b6k/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 0});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({2}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 4){
        return 1;
    }else{
        return 0;
    }
}


int test_promotion_queen_without_eating_black(){
    ChessGameState game;
    game.cod_fem = "8/8/b6k/1q6/8/8/2RPP1p1/2RKN3 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 4});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({62}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -4){
        return 1;
    }else{
        return 0;
    }
}

int test_promotion_queen_eating_white(){
    ChessGameState game;
    game.cod_fem = "3n4/2P5/b6k/1q6/8/5N2/2RPP3/2RK4 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 1});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({3}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 4){
        return 1;
    }else{
        return 0;
    }
}

int test_promotion_queen_eating_black(){
    ChessGameState game;
    game.cod_fem = "8/8/b6k/1q6/8/3N4/2RPP1p1/2RK3R b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 3});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({63}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -4){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_rook_without_eating_white(){
    ChessGameState game;
    game.cod_fem = "7k/2P5/b7/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 69});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({2}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 1 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_rook_without_eating_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/8/2RPP1p1/2R1K3 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 69});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({62}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -1 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_rook_eating_left_white(){
    ChessGameState game;
    game.cod_fem = "1r5k/2P5/b7/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 66});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({1}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 1 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_rook_eating_right_white(){
    ChessGameState game;
    game.cod_fem = "3r3k/2P5/b7/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 72});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({3}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 1 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_rook_eating_left_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/3P4/2R1P1p1/2RK3N b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 66});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({63}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -1 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_rook_eating_right_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/3P4/2R1P1p1/2RK1N2 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 72});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({61}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -1 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_bishop_without_eating_white(){
    ChessGameState game;
    game.cod_fem = "8/2P5/b3k3/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 67});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({2}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 3 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_bishop_without_eating_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/4K3/2RPP1p1/2R5 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 67});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({62}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -3 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_bishop_eating_left_white(){
    ChessGameState game;
    game.cod_fem = "1r6/2P5/b7/1q2k3/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 64});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({1}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 3 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_bishop_eating_right_white(){
    ChessGameState game;
    game.cod_fem = "3r4/2P5/b4k2/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 70});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({3}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 3 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_bishop_eating_left_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/4K3/3P4/2R1P1p1/2R4N b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 64});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({63}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -3 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_bishop_eating_right_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/3P3K/2R1P1p1/2R2N2 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 70});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({61}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -3 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_knight_without_eating_white(){
    ChessGameState game;
    game.cod_fem = "8/2P5/b2k4/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 68});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({2}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 2 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}


int test_subpromotion_knight_without_eating_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/5K2/2RPP1p1/2R5 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 68});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({62}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -2 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_knight_eating_left_white(){
    ChessGameState game;
    game.cod_fem = "1r6/2Pk4/b7/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 65});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({1}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 2 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_knight_eating_right_white(){
    ChessGameState game;
    game.cod_fem = "3r4/2P5/b3k3/1q6/8/8/2R1P3/2R1K3 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 2, 71});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({3}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == 2 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_knight_eating_left_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/3P4/2R1PKp1/2R4N b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 65});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({63}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -2 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_subpromotion_knight_eating_right_black(){
    ChessGameState game;
    game.cod_fem = "7k/8/b7/1q6/8/3P2K1/2R1P1p1/2R2N2 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({6, 6, 71});

    game.set_next_state(action_idx);
    game.set_player();
    int piece_promotion = game.flatten_board.index({61}).item<int>();
    std::cout << game.board << std::endl;

    if (piece_promotion == -2 && game.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_castling_avoided_by_moving_right_rook_white(){
    ChessGameState game;
    game.cod_fem = "r1bq1rk1/ppp2ppp/2np1n2/2b5/2B5/2NP1N2/PPP2PPP/R1BQK2R w KQhq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 7, 6});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 2, 3});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 6, 2});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 3, 7});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    int expected_move = valid_moves_tensor.index({7, 4, 10}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_castling_avoided_by_moving_left_rook_white(){
    ChessGameState game;
    game.cod_fem = "rnbq1rk1/ppppbppp/5n2/4p1B1/8/2NP4/PPPQPPPP/R3KBNR w KQq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 0, 2});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 3, 2});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 1, 6});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 4, 6});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    int expected_move = valid_moves_tensor.index({7, 4, 14}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_castling_avoided_by_moving_right_rook_black(){
    ChessGameState game;
    game.cod_fem = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2NP4/PPP2PPP/R1BQK1NR b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 7, 6});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 3, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 6, 2});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 3, 4});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    int expected_move = valid_moves_tensor.index({0, 4, 10}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_castling_avoided_by_moving_left_rook_black(){
    ChessGameState game;
    game.cod_fem = "r3kbnr/pppqpppp/2np4/8/4P1b1/2N2N2/PPPPBPPP/R1BQ1RK1 b Qkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({0, 0, 2});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 3, 2});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 1, 6});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 4, 6});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    int expected_move = valid_moves_tensor.index({0, 4, 14}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}

int test_repetition_draw_white(){
    ChessGameState game;
    game.cod_fem = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    //primera repeticion
    auto action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    //segunda repetición
    action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    //tercera repetición
    action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_repetition_draw_black(){
    ChessGameState game;
    game.cod_fem = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    //primera repeticion
    auto action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    //segunda repetición
    action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    //tercera repetición
    action_idx = torch::tensor({0, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 0});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({6, 4, 4});

    game.set_next_state(action_idx);
    game.set_player();

    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_check_and_defend_with_check(){
    ChessGameState game;
    game.cod_fem = "3k4/8/1b6/8/8/8/3K4/3R4 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({2, 1, 5});

    game.set_next_state(action_idx);
    game.set_player();
    
    auto check1 = game.on_check;

    action_idx = torch::tensor({6, 3, 2});
    game.set_next_state(action_idx);
    game.set_player();

    auto check2 = game.on_check;
    std::cout << game.board << std::endl;

    if (check1 && check2){
        return 1;
    }else{
        return 0;
    }
}


int test_expected_position_using_instance_copies(){
    /*ChessGameState game;
    game.cod_fem = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({4, 2, 17});

    game.set_next_state(action_idx);
    game.set_player();

    std::cout << game.board << std::endl;

    if (game.on_check){
        return 1;
    }else{
        return 0;
    }*/
    ChessGameState game1;
    //primera repeticion
    auto action_idx = torch::tensor({6, 4, 8});

    game1.set_next_state(action_idx);
    game1.set_player();

    ChessGameState game2(game1);

    action_idx = torch::tensor({1, 4, 12});

    game2.set_next_state(action_idx);
    game2.set_player();

    ChessGameState game3(game2);

    action_idx = torch::tensor({7, 5, 23});

    game3.set_next_state(action_idx);
    game3.set_player();

    ChessGameState game4(game3);

    action_idx = torch::tensor({0, 1, 60});

    game4.set_next_state(action_idx);
    game4.set_player();

    ChessGameState game5(game4);

    action_idx = torch::tensor({7, 1, 57});

    game5.set_next_state(action_idx);
    game5.set_player();

    ChessGameState game6(game5);

    action_idx = torch::tensor({1, 3, 4});

    game6.set_next_state(action_idx);
    game6.set_player();

    ChessGameState game7(game6);

    action_idx = torch::tensor({4, 2, 17});

    game7.set_next_state(action_idx);
    game7.set_player();

    ChessGameState game8(game7);
    
    auto valid_moves_tensor = game8.get_valid_moves();
    auto pair = game8.get_value_and_terminated(valid_moves_tensor);
    std::cout << game8.board << std::endl;

    auto encoded_state = game6.get_encoded_state();
    std::cout << encoded_state.sizes() << std::endl;
    /*for(int i=0;i < 8; ++i){
        auto hist_m_i = encoded_state.index({"...", torch::indexing::Slice(14*i, 14+(14*i))});
        std::cout << hist_m_i.sizes() << std::endl;
        for(int j=0;j < 14; ++j){
            std::cout << hist_m_i.index({"...", j}) << std::endl;
        }
        std::cout << "--------------" << std::endl;
    }*/

    if (pair.first == 0 and !pair.second and game8.on_check){
        return 1;
    }else{
        return 0;
    }
}

int test_draw_king_vs_king_only(){
    ChessGameState game;
    game.cod_fem = "8/4k3/8/8/8/3rK3/8/8 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({5, 4, 6});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_draw_king_bishop_vs_king_only(){
    ChessGameState game;
    game.cod_fem = "8/4k3/8/3r4/2B5/4K3/8/8 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({4, 2, 1});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_draw_king_vs_king_knight_only(){
    ChessGameState game;
    game.cod_fem = "8/1n2k3/8/2R5/8/4K3/8/8 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({1, 1, 60});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}


int test_draw_two_knights_only(){
    ChessGameState game;
    game.cod_fem = "8/1n2k3/8/1n6/8/2R1K3/8/8 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({3, 1, 60});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_draw_bishop_vs_knight_only(){
    ChessGameState game;
    game.cod_fem = "8/1n2k3/8/1n6/8/2R1K3/8/8 b - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({3, 1, 60});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);

    std::cout << game.board << std::endl;

    if (pair.first == 0 and pair.second){
        return 1;
    }else{
        return 0;
    }
}

int test_stuck_pointing_enemy(){
    ChessGameState game;
    game.cod_fem = "rn1qkbnr/ppp3pp/3pbp2/4p3/4P3/1PN5/PBPP1PPP/R2QKBNR w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 3, 25});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({2, 4, 1});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 6, 56});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto expected_move = valid_moves_tensor.index({1,5,11}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 1){
        return 1;
    }else{
        return 0;
    }
}

int test_stuck_same_piece_several_moves(){
    ChessGameState game;
    game.cod_fem = "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({7, 5, 31});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 6, 61});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 4, 10});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    auto expected_move = valid_moves_tensor.index({2,2,60}).item<int>();

    std::cout << game.board << std::endl;

    if (expected_move == 0){
        return 1;
    }else{
        return 0;
    }
}


int test_scape_mate_king_only_move(){
    ChessGameState game;
    game.cod_fem = "4k3/R7/8/8/2Q5/8/8/1K1R4 w - - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    auto action_idx = torch::tensor({4, 2, 14});

    game.set_next_state(action_idx);
    game.set_player();
    
    valid_moves_tensor = game.get_valid_moves();
    int n_valid_moves = torch::sum(valid_moves_tensor).item<int>();

    std::cout << game.board << std::endl;

    if (n_valid_moves == 1){
        return 1;
    }else{
        return 0;
    }
}


int test_error_validation(){
    ChessGameState game;
    game.cod_fem = "rn3b1r/1pp2p1p/p2p1n1q/1k2p1p1/1P1P4/5PPN/PBP1P1BP/R1Q1K2R b KQ - 0 1";
    game.fentoBoard();
    auto valid_moves_tensor = game.get_valid_moves();

    int n_valid_moves = torch::sum(valid_moves_tensor).item<int>();

    std::cout << game.board << std::endl;

    if (n_valid_moves > 0){
        return 1;
    }else{
        return 0;
    }
}


int main() {
    AlphazeroParams alphazero_params;
    alphazero_params.n_iterations = 500;
    alphazero_params.n_selfplay_iterations = 100;
    alphazero_params.num_searches = 800;
    alphazero_params.batch_size = 128;
    alphazero_params.temperature = 0.05;
    alphazero_params.dirichlet_epsilon = 0.25;
    alphazero_params.dirichlet_alpha = 1.0;
    alphazero_params.c_puct = 4.0;
    alphazero_params.num_threads_mcts = 32;


    //ChessGameState game;
    //std::cout << game.board << std::endl;
    /*game.cod_fem = "1r2n1r1/5pp1/R5np/1p1pk3/2Q3Pb/1PP1BB2/1N1RP1K1/8 w HAgb - 0 1";
    game.fentoBoard();
    std::cout << game.board << std::endl;
    game.boardtoFEN();
    std::cout << game.cod_fem << std::endl;
    auto state_encoded = game.get_encoded_state();
    auto valid_moves_tensor = game.get_valid_moves();
    std::cout << valid_moves_tensor.sizes() << std::endl;
    std::cout << "numero de jugadas permitidas: " << torch::sum(valid_moves_tensor).item<int>() << std::endl;
    std::cout << valid_moves_tensor.index({4, 2, 10}) << std::endl;
    auto action_idx = torch::tensor({4, 2, 10});
    game.set_next_state(action_idx);
    std::cout << "Target squares on check: " << game.target_squares_on_check << std::endl;
    std::cout << "Target squares attacked by enemy: " << game.target_squares_attacked_by_enemy << std::endl;
    std::cout << game.board << std::endl;
    game.set_player();
    valid_moves_tensor = game.get_valid_moves();
    auto pair = game.get_value_and_terminated(valid_moves_tensor);
    std::cout << pair << std::endl;

    std::cout << valid_moves_tensor.index({3, 3, 3}) << std::endl;*/

    /*bool terminated = false;
    torch::Tensor valid_moves_tensor = game.get_valid_moves();
    std::pair<int, int> p;
    while(!terminated){
        auto valid_moves_flatten = valid_moves_tensor.view(-1);
        auto valid_moves_aux = torch::zeros(valid_moves_flatten.sizes());
        int action_idx_flat = torch::multinomial(valid_moves_flatten, 1, false).item<int>();
        valid_moves_aux.index_put_({action_idx_flat}, 1);
        valid_moves_aux = valid_moves_aux.view(valid_moves_tensor.sizes());
        auto action_idx = torch::nonzero(valid_moves_aux).squeeze(0);
        std::cout << action_idx << std::endl;

        auto decoded_action = game.decode_action(action_idx);
        std::cout << decoded_action << std::endl;
        game.set_next_state(action_idx);
        std::cout << "Target squares on check: " << game.target_squares_on_check << std::endl;
        std::cout << "Target squares attacked by enemy: " << game.target_squares_attacked_by_enemy << std::endl;
        std::cout << "Celdas clavadas: " << game.squares_on_check_stuck << std::endl;
        game.set_player();
        std::cout << game.board << std::endl;
        valid_moves_tensor = game.get_valid_moves();
        p = game.get_value_and_terminated(valid_moves_tensor);
        terminated = p.second;
        game.boardtoFEN();
        std::cout << game.cod_fem << std::endl;
        if(game.on_check){
            std::cout << "Check!" << std::endl;
        }

        torch::Tensor mask1 = (game.board == -5).to(torch::kInt16).unsqueeze(-1);
        torch::Tensor mask2 = (game.board == 5).to(torch::kInt16).unsqueeze(-1);
        int exist_king1 = torch::sum(mask1).item<int>();
        int exist_king2 = torch::sum(mask2).item<int>();
        if(exist_king1 == 0 || exist_king2 == 0){
            std::cout << "Error por haberse comido un rey!" << std::endl;
            break;
        }
    }
    std::cout << "Terminado por: " << p << std::endl;*/
    int total_results = 0;
    int total_tests = 0;

    int result = test_double_check_mate_white();
    total_results += result;
    total_tests++;
    result = test_double_check_mate_black();
    total_results += result;
    total_tests++;
    result = test_double_check_defense_white();
    total_results += result;
    total_tests++;
    result = test_double_check_defense_black();
    total_results += result;
    total_tests++;
    result = test_drowning_pawns_white();
    total_results += result;
    total_tests++;
    result = test_drowning_pawns_black();
    total_results += result;
    total_tests++;
    result = test_drowning_stuck_pieces_white();
    total_results += result;
    total_tests++;
    result = test_drowning_simple_black();
    total_results += result;
    total_tests++;
    result = test_knight_only_check_mate_white();
    total_results += result;
    total_tests++;
    result = test_knight_only_check_mate_black();
    total_results += result;
    total_tests++;
    result = test_knight_only_defense_check_mate_black();
    total_results += result;
    total_tests++;
    result = test_knight_only_defense_check_mate_white();
    total_results += result;
    total_tests++;
    result = test_large_castling_doing_check();
    total_results += result;
    total_tests++;
    result = test_short_castling_white();
    total_results += result;
    total_tests++;
    result = test_long_castling_white();
    total_results += result;
    total_tests++;
    result = test_short_castling_black();
    total_results += result;
    total_tests++;
    result = test_long_castling_black();
    total_results += result;
    total_tests++;
    result = test_enPassant_white();
    total_results += result;
    total_tests++;
    result = test_enPassant_black();
    total_results += result;
    total_tests++;
    result = test_cut_castling_short_white();
    total_results += result;
    total_tests++;
    result = test_cut_castling_long_white();
    total_results += result;
    total_tests++;
    result = test_cut_castling_short_black();
    total_results += result;
    total_tests++;
    result = test_cut_castling_long_black();
    total_results += result;
    total_tests++;
    result = test_cut_castling_by_check();
    total_results += result;
    total_tests++;
    result = test_avoided_by_moving_king_white();
    total_results += result;
    total_tests++;
    result = test_avoided_by_moving_king_black();
    total_results += result;
    total_tests++;
    result = test_stuck_piece_white();
    total_results += result;
    total_tests++;
    result = test_stuck_piece_black();
    total_results += result;
    total_tests++;
    result = test_promotion_queen_without_eating_white();
    total_results += result;
    total_tests++;
    result = test_promotion_queen_without_eating_black();
    total_results += result;
    total_tests++;
    result = test_promotion_queen_eating_white();
    total_results += result;
    total_tests++;
    result = test_promotion_queen_eating_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_rook_without_eating_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_rook_without_eating_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_rook_eating_left_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_rook_eating_right_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_rook_eating_left_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_rook_eating_right_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_bishop_without_eating_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_bishop_without_eating_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_bishop_eating_left_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_bishop_eating_right_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_bishop_eating_left_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_bishop_eating_right_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_knight_without_eating_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_knight_without_eating_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_knight_eating_left_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_knight_eating_right_white();
    total_results += result;
    total_tests++;
    result = test_subpromotion_knight_eating_left_black();
    total_results += result;
    total_tests++;
    result = test_subpromotion_knight_eating_right_black();
    total_results += result;
    total_tests++;
    result = test_castling_avoided_by_moving_right_rook_white();
    total_results += result;
    total_tests++;
    result = test_castling_avoided_by_moving_left_rook_white();
    total_results += result;
    total_tests++;
    result = test_castling_avoided_by_moving_right_rook_black();
    total_results += result;
    total_tests++;
    result = test_castling_avoided_by_moving_left_rook_black();
    total_results += result;
    total_tests++;
    result = test_repetition_draw_white();
    total_results += result;
    total_tests++;
    result = test_repetition_draw_black();
    total_results += result;
    total_tests++;
    result = test_check_and_defend_with_check();
    total_results += result;
    total_tests++;
    result = test_expected_position_using_instance_copies();
    total_results += result;
    total_tests++;
    result = test_draw_king_vs_king_only();
    total_results += result;
    total_tests++;
    result = test_draw_king_bishop_vs_king_only();
    total_results += result;
    total_tests++;
    result = test_draw_king_vs_king_knight_only();
    total_results += result;
    total_tests++;
    result = test_draw_two_knights_only();
    total_results += result;
    total_tests++;
    result = test_stuck_pointing_enemy();
    total_results += result;
    total_tests++;
    result = test_stuck_same_piece_several_moves();
    total_results += result;
    total_tests++;
    result = test_scape_mate_king_only_move();
    total_results += result;
    total_tests++;
    result = test_error_validation();
    total_results += result;
    total_tests++;

    if(total_results == total_tests){
        std::cout << "Tests pasados!" << std::endl;
    }else{
        std::cout << "Tests fallados!" << total_results << "--" << total_tests << std::endl;
    }

    ChessGameState game;
    game.cod_fem = "r1bqk1nr/ppp2ppp/2np4/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1";
    game.fentoBoard();

    auto tiempo_inicio = std::chrono::high_resolution_clock::now();

    auto v = game.get_valid_moves();

    auto tiempo_fin = std::chrono::high_resolution_clock::now();
    auto duracion = std::chrono::duration_cast<std::chrono::microseconds>(tiempo_fin - tiempo_inicio);

    // Convierte la duración a segundos
    double segundos = static_cast<double>(duracion.count()) / 1'000'000.0;

    int n_valid_moves = torch::sum(v).item<int>();

    std::cout << "N jugadas válidas: " << n_valid_moves << std::endl;

    std::cout << "Tiempo gastado en tomar jugadas válidas: " << segundos << std::endl;

    /*torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << device << std::endl;

    int batch_size = 8;

    ChessGameState game;

    std::shared_ptr<BufferEvalManager> buffer_eval_manager = std::make_shared<BufferEvalManager>(device, batch_size);
    buffer_eval_manager->set_model("../models/model_0.pt");
    std::thread controller_thread([buffer_eval_manager] { buffer_eval_manager->buffer_eval_controller(); });*/
    /*MCTS mcts(alphazero_params, device, buffer_eval_manager);

    mcts.reset_game(false);

    torch::Tensor action_probs = mcts.search(false);

    int action_idx = torch::multinomial(action_probs, 1, false).item<int>();
    auto policy_zeros = torch::zeros(game.action_size_flat);
    policy_zeros.index_put_({action_idx}, 1);
    policy_zeros = policy_zeros.view(game.action_size);
    torch::Tensor action_idx_tensor = torch::nonzero(policy_zeros).squeeze(0);

    std::string action_idx_str = game.tensorToString(action_idx_tensor);
    mcts.update_root(action_idx_str);

    game.set_next_state(action_idx_tensor);
    game.set_player();

    action_probs = mcts.search(false);

    action_idx = torch::multinomial(action_probs, 1, false).item<int>();
    policy_zeros = torch::zeros(game.action_size_flat);
    policy_zeros.index_put_({action_idx}, 1);
    policy_zeros = policy_zeros.view(game.action_size);
    action_idx_tensor = torch::nonzero(policy_zeros).squeeze(0);

    action_idx_str = game.tensorToString(action_idx_tensor);
    mcts.update_root(action_idx_str);

    game.set_next_state(action_idx_tensor);
    game.set_player();

    std::cout << game.board << std::endl;*/

    //primera repeticion
    /*auto action_idx = torch::tensor({6, 4, 8});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({1, 4, 12});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({7, 5, 23});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 5, 21});

    game.set_next_state(action_idx);
    game.set_player();

    //segunda repetición
    action_idx = torch::tensor({7, 3, 9});

    game.set_next_state(action_idx);
    game.set_player();

    action_idx = torch::tensor({0, 1, 60});

    game.set_next_state(action_idx);
    game.set_player();

    std::cout << game.board <<std::endl;

    auto input_state = game.get_encoded_state();

    std::promise< std::pair<torch::Tensor, float> > promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> guard(buffer_eval_manager->buffer_eval_mutex);
        buffer_eval_manager->buffer_eval_inputs.push_back(std::move(input_state));
        buffer_eval_manager->buffer_eval_promises.push_back(std::move(promise));
        if (buffer_eval_manager->buffer_eval_inputs.size() >= buffer_eval_manager->batch_size) {
            buffer_eval_manager->buffer_eval_cond.notify_one();
        }
    }

    auto prediction = future.get();
    auto policy = prediction.first.view(game.action_size);

    auto prob_checkmate = policy.index({5, 5, 24});
    std::cout << "Prob easy checkmate: " << prob_checkmate << std::endl;
    auto max_index_flat = torch::argmax(policy);
    int64_t index = max_index_flat.item<int64_t>();
    int D1 = policy.size(0);
    int D2 = policy.size(1);
    int D3 = policy.size(2);

    int idx_D1 = index / (D2 * D3); // Índice en la primera dimensión
    int idx_D2 = (index % (D2 * D3)) / D3; // Índice en la segunda dimensión
    int idx_D3 = index % D3;

    std::cout << "Índices 3D: (" << idx_D1 << ", " << idx_D2 << ", " << idx_D3 << ")" << std::endl;

    auto prob_best_move = policy.index({idx_D1, idx_D2, idx_D3});
    std::cout << "Prob best move: " << prob_best_move << std::endl;

    auto tiempo_inicio = std::chrono::high_resolution_clock::now();

    auto v = game.get_valid_moves();

    auto tiempo_fin = std::chrono::high_resolution_clock::now();
    auto duracion = std::chrono::duration_cast<std::chrono::microseconds>(tiempo_fin - tiempo_inicio);

    // Convierte la duración a segundos
    double segundos = static_cast<double>(duracion.count()) / 1'000'000.0;

    int n_valid_moves = torch::sum(v).item<int>();

    std::cout << "N jugadas válidas: " << n_valid_moves << std::endl;

    std::cout << "Tiempo gastado en tomar jugadas válidas: " << segundos << std::endl;

    tiempo_inicio = std::chrono::high_resolution_clock::now();

    auto p = game.get_value_and_terminated(v);

    tiempo_fin = std::chrono::high_resolution_clock::now();
    duracion = std::chrono::duration_cast<std::chrono::microseconds>(tiempo_fin - tiempo_inicio);

    segundos = static_cast<double>(duracion.count()) / 1'000'000.0;

    std::cout << "Tiempo gastado en ver si la partida ha terminado: " << segundos << std::endl;

    action_idx = torch::tensor({5, 5, 24});

    tiempo_inicio = std::chrono::high_resolution_clock::now();

    game.set_next_state(action_idx);

    tiempo_fin = std::chrono::high_resolution_clock::now();
    duracion = std::chrono::duration_cast<std::chrono::microseconds>(tiempo_fin - tiempo_inicio);

    segundos = static_cast<double>(duracion.count()) / 1'000'000.0;

    std::cout << "Tiempo gastado en hacer una acción: " << segundos << std::endl;

    buffer_eval_manager->terminate_thread_buffer_eval = true;
    if (controller_thread.joinable()) {
        controller_thread.join(); // Espera a que el hilo controlador termine
    }*/

    //auto maxIter = std::max_element(mcts.initial_state_node->n_children.begin(), mcts.initial_state_node->n_children.end());

    // Desreferencia el iterador para obtener el valor máximo.
    //int maxValor = *maxIter;
    //std::cout << "Máximo número de hijos: " << maxValor << std::endl;
    //std::cout << "Nodos creados: " << mcts.initial_state_node->contador_creaciones << std::endl;
    //mcts.clean_tree();
    //std::cout << "Nodos eliminados: " << mcts.initial_state_node->contador_eliminaciones << std::endl;

    return 0;
}