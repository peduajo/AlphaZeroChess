#pragma once

#include <iostream>
#include <torch/torch.h>
#include <utility>

class ChessGameState {
public:
    static int row_count;
    static int column_count;
    static int action_planes;
    static std::vector<int64_t> action_size;
    static std::vector<int64_t> action_size_flat;
    static std::unordered_map<char, int> pieceToNumber;
    static std::unordered_map<int, char> numberToPiece;
    static std::unordered_map<int, char> colNumToColChar;
    static std::unordered_map<int, char> numberToPieceDecode;
    static std::unordered_map<char, int> colCharToColNum;
    torch::Tensor board; // Representación del tablero
    torch::Tensor flatten_board;
    int currentPlayer; // 1 para blancas, 0 para negras
    std::string castlingRigtsWhite; // Derechos de enroque, por ejemplo "KQ"
    std::string castlingRigtsBlack;
    int enPassantTarget; // Posible captura al paso, -1 no es posible o sino se pone el target_square
    int halfMoveClock; // Contador de medio movimientos para la regla de los 50 movimientos
    static int limit_plays_draw;
    int fullMoveNumber; // Número de movimientos completos
    std::string cod_fem;
    bool on_check; //significa que el jugador actual está en jaque o que su rey está amenazado
    bool on_double_check; //significa que el jugador actual está sobre jaque doble y su única opción es movel el rey
    int count_checks_pieces;

    torch::Tensor valid_moves_cache;

    static std::set<int> sliding_pieces;
    static std::vector<int> directions_pawn_white;
    static std::vector<int> directions_pawn_black;
    //esto de target squares castling tiene los indices de las celdas que hay que comprobar si están atacadas o hay fichas para el movimiento del enroque en cada dirección para cada color
    static std::vector<int> target_squares_white_castling_kside;
    static std::vector<int> target_squares_white_castling_qside;
    static std::vector<int> target_squares_black_castling_kside;
    static std::vector<int> target_squares_black_castling_qside;
    static std::vector<int> knight_bishop_pieces;
    std::set<int> target_squares_attacked_by_enemy; //creada al realizar una jugada para que el contrario lo tenga en cuenta
    //std::set<int> squares_on_check_stuck; //lista de celdas con piezas que están clavadas y no se pueden mover
    std::set<int> target_squares_on_check; //lista de celdas teniendo todas las celdas de rango que provocan jaque incluyendo la pieza que lo provoca
    static std::unordered_map<int, std::unordered_map<int, int>> sliding_pieces_to_edge_dict;
    static std::unordered_map<int, std::unordered_map<int, int>> initialize_sliding_pieces_to_edge_dict();

    static std::unordered_map<int, std::unordered_map<int, bool>> knight_pieces_to_edge_dict;
    static std::unordered_map<int, std::unordered_map<int, bool>> initialize_knight_pieces_to_edge_dict();

    static std::unordered_map<int, std::unordered_map<int, bool>> pawn_pieces_to_edge_dict;
    static std::unordered_map<int, std::unordered_map<int, bool>> initialize_pawn_pieces_to_edge_dict();

    static std::unordered_map<int, std::unordered_map<int, bool>> king_piece_to_edge_dict;
    static std::unordered_map<int, std::unordered_map<int, bool>> initialize_king_pieces_to_edge_dict();

    static std::unordered_map<int, int> get_direction_offsets_map(std::string mode);
    static std::unordered_map<int, int> sliding_direction_offsets;
    static std::unordered_map<int, int> knight_direction_offsets;

    static torch::Tensor idx_subpromotions;
    static std::vector<int> order_m_planes_white;
    static std::vector<int> order_m_planes_black;

    std::unordered_map<std::string, int> repetitions_per_position;
    std::vector<torch::Tensor> history_m;
    static int m_features;
    static int t_timesteps;

    static std::vector<std::pair<torch::Tensor, torch::Tensor>> positions_data;
    static std::mutex positionsMutex;

    // Constructor para inicializar el estado del juego con valores por defecto
    ChessGameState(){
        this->reset();
    }

    ChessGameState(const ChessGameState& father)
        //los tensores de libtorch hay que clonarlos
        :   board(father.board.clone()),
            flatten_board(father.flatten_board.clone()),
            currentPlayer(father.currentPlayer),
            castlingRigtsWhite(father.castlingRigtsWhite),
            castlingRigtsBlack(father.castlingRigtsBlack),
            enPassantTarget(father.enPassantTarget),
            halfMoveClock(father.halfMoveClock),
            fullMoveNumber(father.fullMoveNumber),
            on_check(father.on_check),
            on_double_check(father.on_double_check),
            count_checks_pieces(father.count_checks_pieces),
            target_squares_attacked_by_enemy(father.target_squares_attacked_by_enemy),
            //squares_on_check_stuck(father.squares_on_check_stuck),
            target_squares_on_check(father.target_squares_on_check),
            repetitions_per_position(father.repetitions_per_position){
        
        for(const torch::Tensor &m: father.history_m){
            history_m.push_back(m.clone());
        }
    }

    void boardtoFEN();
    void fentoBoard();
    void reset();
    torch::Tensor get_encoded_state();
    torch::Tensor get_encoded_state_vm();
    float get_opponent_value(float value);
    torch::Tensor get_valid_moves();
    void set_valid_moves_piece(int piece_idx, int start_square, torch::Tensor& valid_moves_tensor);
    std::vector<int> set_valid_moves_sliding_piece(int start_square, int abs_piece, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves);
    std::vector<int> set_valid_moves_knight_piece(int start_square, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves);
    std::vector<int> set_valid_moves_pawn_piece(int start_square, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves);
    std::vector<int> set_valid_moves_king_piece(int start_square, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves);
    void set_valid_moves_castling(int direction_idx, int col, int row, int idx_move_target, torch::Tensor& valid_moves_tensor, std::vector<int>& target_squares_castling);
    bool on_check_control_move(int target_square);
    void set_next_state(const torch::Tensor& action_idx);
    std::tuple<int, int, bool> set_next_state_pawn(int direction, int start_square, int row, int piece_idx, bool set_board);
    std::pair<int, int> set_next_state_king(int start_square, int direction, int piece_idx, bool set_board);
    void set_target_squares_attacked();
    std::vector<int> get_target_squares_piece(int piece_idx, int start_square);
    void set_enemy_pieces_on_check_stuck();
    std::pair<int, bool> get_value_and_terminated(const torch::Tensor& valid_moves_tensor);
    void set_player();
    std::string decode_action(const torch::Tensor& action_idx);
    std::string int_square_to_string(int target_square);
    std::string tensorToString(const torch::Tensor& tensor);
    bool check_enough_pieces();
    static void addPosition(torch::Tensor input, torch::Tensor output);
    bool check_validation_move(const std::vector<int>& action_idx);
    bool valid_king_on_check(const torch::Tensor& flatten_board_aux, int king_square);
    torch::Tensor simulate_next_state_pawn(int direction, int start_square, int row, int piece_idx, torch::Tensor& flatten_board_aux);
};