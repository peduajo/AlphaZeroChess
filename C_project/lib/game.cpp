#include <iostream>
#include <torch/torch.h>
#include <utility>
#include "game.h"
#include <cstdlib>
#include <sstream>

int ChessGameState::row_count = 8;
int ChessGameState::column_count = 8;
int ChessGameState::action_planes = 73;
int ChessGameState::m_features = 14;
int ChessGameState::t_timesteps = 8;
int ChessGameState::limit_plays_draw = 50;
std::vector<int64_t> ChessGameState::action_size = {static_cast<int64_t>(row_count), static_cast<int64_t>(column_count), 73};
std::vector<int64_t> ChessGameState::action_size_flat = {static_cast<int64_t>(row_count)*static_cast<int64_t>(column_count)*73};
std::unordered_map<char, int> ChessGameState::pieceToNumber = {{'r', -1}, {'n', -2}, {'b', -3}, {'q', -4}, {'k', -5}, {'p', -6},
                                                               {'R', 1}, {'N', 2}, {'B', 3}, {'Q', 4}, {'K', 5}, {'P', 6}};
std::unordered_map<int, char> ChessGameState::numberToPiece = {{-1, 'r'}, {-2, 'n'}, {-3, 'b'}, {-4, 'q'}, {-5, 'k'}, {-6, 'p'},
                                                               {1, 'R'}, {2, 'N'}, {3, 'B'}, {4, 'Q'}, {5, 'K'}, {6, 'P'}};
std::unordered_map<int, char> ChessGameState::colNumToColChar = {{0, 'a'}, {1, 'b'}, {2, 'c'}, {3, 'd'}, {4, 'e'}, {5, 'f'}, {6, 'g'}, {7, 'h'}};
std::unordered_map<char, int> ChessGameState::colCharToColNum = {{'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5}, {'g', 6}, {'h', 7}};
std::unordered_map<int, char> ChessGameState::numberToPieceDecode = {{1,'T'}, {2, 'C'}, {3, 'A'}, {4, 'D'}, {5, 'R'}};
std::set<int> ChessGameState::sliding_pieces = {-1, 1, -3, 3, -4, 4};
std::vector<int> ChessGameState::directions_pawn_white = {7, 0, 1};
std::vector<int> ChessGameState::directions_pawn_black = {3, 4, 5};
std::vector<int> ChessGameState::target_squares_white_castling_kside = {61, 62};
std::vector<int> ChessGameState::target_squares_white_castling_qside = {58, 59};
std::vector<int> ChessGameState::target_squares_black_castling_kside = {5, 6};
std::vector<int> ChessGameState::target_squares_black_castling_qside = {2, 3};
torch::Tensor ChessGameState::idx_subpromotions = torch::tensor({64, 65, 66});
std::vector<int> ChessGameState::order_m_planes_white = {1,2,3,4,5,6,-1,-2,-3,-4,-5,-6};
std::vector<int> ChessGameState::order_m_planes_black = {-1,-2,-3,-4,-5,-6,1,2,3,4,5,6};
std::vector<int> ChessGameState::knight_bishop_pieces = {2, -2, 3, -3};

std::unordered_map<int, std::unordered_map<int, int>> ChessGameState::initialize_sliding_pieces_to_edge_dict(){
    std::unordered_map<int, std::unordered_map<int, int>> sliding_pieces_to_edge_dict;

    int offset;
    for(int row = 0; row < row_count; ++row){
        for(int col = 0; col < column_count; ++col){
            int square_idx = row_count*row + col;
            for(int direction_idx=0; direction_idx < 8; ++direction_idx){
                switch(direction_idx){
                    case 0:
                        //norte
                        offset = row;
                        break;
                    //noreste
                    case 1:
                        //es el minimo de norte y este
                        offset = std::min(row, column_count - 1 - col);
                        break;
                    //este
                    case 2:
                        offset = column_count - 1 - col;
                        break;
                    //sureste
                    case 3:
                        //es el minimo de este y sur
                        offset = std::min(row_count - 1 - row, column_count - 1 - col);
                        break;
                    //sur
                    case 4:
                        offset = row_count - 1 - row;
                        break;
                    //suroeste
                    case 5:
                        //es el minimo de sur y oeste
                        offset = std::min(row_count - 1 - row, col);
                        break;
                    //oeste
                    case 6:
                        offset = col;
                        break;
                    //noroeste
                    case 7:
                        //es el minimo de norte y este
                        offset = std::min(row, col);
                        break;
                }   
                sliding_pieces_to_edge_dict[square_idx][direction_idx] = offset;
            }
        }
    }

    return sliding_pieces_to_edge_dict;
}

std::unordered_map<int, std::unordered_map<int, int>> ChessGameState::sliding_pieces_to_edge_dict = ChessGameState::initialize_sliding_pieces_to_edge_dict();

std::unordered_map<int, std::unordered_map<int, bool>> ChessGameState::initialize_knight_pieces_to_edge_dict(){
    std::unordered_map<int, std::unordered_map<int, bool>> knight_pieces_to_edge_dict;

    bool offset_allowed;
    for(int row = 0; row < row_count; ++row){
        for(int col = 0; col < column_count; ++col){
            int square_idx = row_count*row + col;
            for(int direction_idx=0; direction_idx < 8; ++direction_idx){
                switch(direction_idx){
                    case 0:
                        offset_allowed = (row > 1 && col > 0) ? true : false;
                        break;
                    case 1:
                        offset_allowed = (row > 1 && col < 7) ? true : false;
                        break;
                    case 2:
                        offset_allowed = (row > 0 && col < 6) ? true : false;
                        break;
                    case 3:
                        offset_allowed = (row < 7 && col < 6) ? true : false;
                        break;
                    case 4:
                        offset_allowed = (row < 6 && col < 7) ? true : false;
                        break;
                    case 5:
                        offset_allowed = (row < 6 && col > 0) ? true : false;
                        break;
                    case 6:
                        offset_allowed = (row < 7 && col > 1) ? true : false;
                        break;
                    case 7:
                        offset_allowed = (row > 0 && col > 1) ? true : false;
                        break;
                }   
                knight_pieces_to_edge_dict[square_idx][direction_idx] = offset_allowed;
            }
        }
    }

    return knight_pieces_to_edge_dict;
}

std::unordered_map<int, std::unordered_map<int, bool>> ChessGameState::knight_pieces_to_edge_dict = ChessGameState::initialize_knight_pieces_to_edge_dict();

std::unordered_map<int, std::unordered_map<int, bool>> ChessGameState::initialize_pawn_pieces_to_edge_dict(){
    std::unordered_map<int, std::unordered_map<int, bool>> pawn_pieces_to_edge_dict;

    bool offset_allowed;
    for(int row = 0; row < row_count; ++row){
        for(int col = 0; col < column_count; ++col){
            int square_idx = row_count*row + col;
            for(int direction_idx=0; direction_idx < 8; ++direction_idx){
                switch(direction_idx){
                    case 7:
                        offset_allowed = (col > 0) ? true : false;
                        break;
                    case 0:
                        offset_allowed = true;
                        break;
                    case 1:
                        offset_allowed = (col < 7) ? true : false;
                        break;
                    case 3:
                        offset_allowed = (col < 7) ? true : false;
                        break;
                    case 4:
                        offset_allowed = true;
                        break;
                    case 5:
                        offset_allowed = (col > 0) ? true : false;
                        break;
                    default:
                        offset_allowed = false;
                }   
                pawn_pieces_to_edge_dict[square_idx][direction_idx] = offset_allowed;
            }
        }
    }

    return pawn_pieces_to_edge_dict;
}

std::unordered_map<int, std::unordered_map<int, bool>> ChessGameState::pawn_pieces_to_edge_dict = ChessGameState::initialize_pawn_pieces_to_edge_dict();

std::unordered_map<int, std::unordered_map<int, bool>> ChessGameState::initialize_king_pieces_to_edge_dict(){
    std::unordered_map<int, std::unordered_map<int, bool>> king_piece_to_edge_dict;

    bool offset_allowed;
    for(int row = 0; row < row_count; ++row){
        for(int col = 0; col < column_count; ++col){
            int square_idx = row_count*row + col;
            for(int direction_idx=0; direction_idx < 8; ++direction_idx){
                switch(direction_idx){
                    case 0:
                        offset_allowed = (row > 0) ? true : false;
                        break;
                    case 1:
                        offset_allowed = (row > 0 && col < 7) ? true : false;
                        break;
                    case 2:
                        offset_allowed = (col < 7) ? true : false;
                        break;
                    case 3:
                        offset_allowed = (row < 7 && col < 7) ? true : false;
                        break;
                    case 4:
                        offset_allowed = (row < 7) ? true : false;
                        break;
                    case 5:
                        offset_allowed = (row < 7 && col > 0) ? true : false;
                        break;
                    case 6:
                        offset_allowed = (col > 0) ? true : false;
                        break;
                    case 7:
                        offset_allowed = (row > 0 && col > 0) ? true : false;
                        break;
                }   
                king_piece_to_edge_dict[square_idx][direction_idx] = offset_allowed;
            }
        }
    }

    return king_piece_to_edge_dict;
}

std::unordered_map<int, std::unordered_map<int, bool>> ChessGameState::king_piece_to_edge_dict = ChessGameState::initialize_king_pieces_to_edge_dict();

std::unordered_map<int, int> ChessGameState::get_direction_offsets_map(std::string mode){
    std::unordered_map<int, int> direction_offsets;
    if(mode == "sliding"){
        direction_offsets[0] = -row_count;
        direction_offsets[1] = -row_count+1;
        direction_offsets[2] = 1;
        direction_offsets[3] = row_count+1;
        direction_offsets[4] = row_count;
        direction_offsets[5] = row_count-1;
        direction_offsets[6] = -1;
        direction_offsets[7] = -row_count-1;
    }else if(mode=="knight"){
        direction_offsets[0] = -2*row_count-1;
        direction_offsets[1] = -2*row_count+1;
        direction_offsets[2] = -row_count+2;
        direction_offsets[3] = row_count+2;
        direction_offsets[4] = 2*row_count+1;
        direction_offsets[5] = 2*row_count-1;
        direction_offsets[6] = row_count - 2;
        direction_offsets[7] = -row_count - 2;
    }
    return direction_offsets;
}

std::unordered_map<int, int> ChessGameState::sliding_direction_offsets = ChessGameState::get_direction_offsets_map("sliding");
std::unordered_map<int, int> ChessGameState::knight_direction_offsets = ChessGameState::get_direction_offsets_map("knight");

std::vector<std::pair<torch::Tensor, torch::Tensor>> ChessGameState::positions_data;
std::mutex ChessGameState::positionsMutex;

void ChessGameState::addPosition(torch::Tensor input, torch::Tensor output) {
    std::lock_guard<std::mutex> lock(positionsMutex); // Bloquea el mutex durante la duración del scope

    auto p = std::make_pair(input.squeeze(0).to(torch::kInt8), output.squeeze(0).to(torch::kInt8));
    positions_data.push_back(p);
}

std::string ChessGameState::tensorToString(const torch::Tensor& tensor) {
    std::stringstream ss;
    ss << tensor;
    return ss.str();
}


std::pair<int, bool> ChessGameState::get_value_and_terminated(const torch::Tensor& valid_moves_tensor){
    int n_moves_available = torch::sum(valid_moves_tensor).item<int>();
    bool no_moves_condition = n_moves_available == 0;

    auto encoded_position = tensorToString(flatten_board);
    bool position_repeated = repetitions_per_position[encoded_position] == 3;
    bool no_eat_limit = halfMoveClock >= limit_plays_draw;
    bool enough_pieces = check_enough_pieces();

    if(n_moves_available == 0 && on_check){
        return std::pair<int, bool>(1, true);
    }
    if (no_moves_condition || position_repeated || no_eat_limit || !enough_pieces){
        return std::pair<int, bool>(0, true);
    }
    return std::pair<int, bool>(0, false);
}


bool ChessGameState::check_enough_pieces(){
    bool enough = true;
    torch::Tensor mask = (flatten_board != 0).to(torch::kInt16);
    int n_pieces = torch::sum(mask).item<int>();
    if(n_pieces == 2){
        enough = false;
    }else if(n_pieces == 3){
        //comprobar casos de caballo/alfil+rey vs rey
        for(int piece_idx:knight_bishop_pieces){
            torch::Tensor mask_p = (flatten_board == piece_idx).to(torch::kInt16);
            int n_pieces_p = torch::sum(mask_p).item<int>();
            if(n_pieces_p == 1){
                enough = false;
            }
        }
    }else if(n_pieces == 4){
        torch::Tensor mask_w = (flatten_board == 2).to(torch::kInt16);
        torch::Tensor mask_b = (flatten_board == -2).to(torch::kInt16);

        int n_pieces_w = torch::sum(mask_w).item<int>();
        int n_pieces_b = torch::sum(mask_b).item<int>();

        if(n_pieces_w == 2 || n_pieces_b == 2){
            enough = false;
        }

    }
    return enough;
}


void ChessGameState::set_player(){
    currentPlayer = (currentPlayer == 1) ? 0 : 1;
}


void ChessGameState::set_next_state(const torch::Tensor& action_idx){
    //según la acción se modifica el tablero
    //this->modifyBoard(action_idx);
    //modificamos las variables propias del juego después de realizar la acción:
    //on_check -- OK
    //on_double_check -- OK
    //enPassantTarget -- OK
    //castlingRightsWhite/Black -- OK
    //repetitions_per_position -- OK
    //history_m -- OK
    //target_squares_on_check -- OK
    //squares_on_check_stuck -- OK
    //target_squares_attacked_by_enemy -- OK
    //halfMoveClock (se aumenta si no se come ninguna pieza) -- OK

    //decode action tensor into dictionary decoded
    int row = action_idx.index({0}).item<int>();
    int col = action_idx.index({1}).item<int>();
    int direction = action_idx.index({2}).item<int>();

    int start_square = row*row_count + col;
    int piece_idx = flatten_board.index({start_square}).item<int>();
    int abs_piece = abs(piece_idx);

    //se deja sin pieza siempre la celda de origen
    flatten_board.index_put_({start_square}, 0);
    int target_piece;
    bool new_en_passant = false;

    //dependiendo la pieza que sea se hace el movimiento
    //caso reina, alfil, torre
    if(sliding_pieces.find(piece_idx) != sliding_pieces.end()) {
        int n = direction / column_count;

        int direction_idx = direction % column_count;
        int offset_dir = sliding_direction_offsets[direction_idx];
        int target_square = start_square + offset_dir * (n + 1);

        target_piece = flatten_board.index({target_square}).item<int>();

        flatten_board.index_put_({target_square}, piece_idx);
        //caso de mover torre y permisos de enroque al moverla
        if(abs_piece == 1){
            if(currentPlayer == 1 && castlingRigtsWhite.length() > 0){
                if(start_square == 56){
                    castlingRigtsWhite.erase(std::remove(castlingRigtsWhite.begin(), castlingRigtsWhite.end(), 'Q'), castlingRigtsWhite.end());
                }else if(start_square == 63){
                    castlingRigtsWhite.erase(std::remove(castlingRigtsWhite.begin(), castlingRigtsWhite.end(), 'K'), castlingRigtsWhite.end());
                }
            }else if(currentPlayer == 0 && castlingRigtsBlack.length() > 0){
                if(start_square == 0){
                    castlingRigtsBlack.erase(std::remove(castlingRigtsBlack.begin(), castlingRigtsBlack.end(), 'q'), castlingRigtsBlack.end());
                }else if(start_square == 7){
                    castlingRigtsBlack.erase(std::remove(castlingRigtsBlack.begin(), castlingRigtsBlack.end(), 'k'), castlingRigtsBlack.end());
                }
            }
        }

    //tratamiento de caballos
    }else if(abs_piece == 2){
        int direction_idx = direction - 56;
        int target_square = start_square + knight_direction_offsets[direction_idx];

        target_piece = flatten_board.index({target_square}).item<int>();

        flatten_board.index_put_({target_square}, piece_idx);        
    //tratamiento de peones, incluida la coronación 
    }else if(abs_piece == 6){
        auto info_target = this->set_next_state_pawn(direction, start_square, row, piece_idx, true);
        new_en_passant = std::get<2>(info_target);
    //tratamiento de rey, incluido el enroque
    }else if (abs_piece == 5){
        auto info_target = this->set_next_state_king(start_square, direction, piece_idx, true);
        target_piece = info_target.second;
    }

    if (!new_en_passant){
        enPassantTarget = -1;
    }

    //set haldMoveClock
    if (target_piece == 0){
        halfMoveClock++;
    }else if(target_piece != 0 || abs_piece == 6){
        halfMoveClock = 0;
    }

    //add repetition position
    auto encoded_position = tensorToString(flatten_board);
    auto it = repetitions_per_position.find(encoded_position);
    int current_repetitions = 0;
    if(it != repetitions_per_position.end()){
        // El elemento existe, incrementa su conteo
        current_repetitions = repetitions_per_position[encoded_position];
        repetitions_per_position[encoded_position] = current_repetitions + 1;
    } else {
        // El elemento no existe, inicialízalo con 1 porque es la primera ocurrencia
        repetitions_per_position[encoded_position] = 1;
    }

    std::vector<torch::Tensor> board_m;
    torch::Tensor board_repetitions = torch::zeros({8,8,2});
    if(current_repetitions == 1){
        board_repetitions.index_put_({"...", 0}, 1);
    }else if(current_repetitions == 2){
        board_repetitions.index_put_({"...", 1}, 1);
    }
    board_m.push_back(board_repetitions);

    board = flatten_board.view({row_count, column_count});
    board_m.push_back(board.unsqueeze(-1));

    auto state_m = torch::concat(board_m, 2);

    history_m.erase(history_m.begin());
    history_m.push_back(state_m);

    this->set_target_squares_attacked();
}


void ChessGameState::set_enemy_pieces_on_check_stuck(){
    //locate enemy king square
    int enemy_king_piece_idx = 5;
    if(currentPlayer == 1){
        enemy_king_piece_idx = -5;
    }

    torch::Tensor mask = (flatten_board == enemy_king_piece_idx).to(torch::kInt16);
    int enemy_king_target_square = torch::argmax(mask).item<int>();

    for(int direction_idx=0; direction_idx < 8; ++direction_idx){
        bool is_even = direction_idx % 2 == 0;

        int offset_dir = sliding_direction_offsets[direction_idx]; //offset de squares por celda
        int offset_limit = sliding_pieces_to_edge_dict[enemy_king_target_square][direction_idx];

        int n_straight_allies = 0; //aliados respecto al rey enemigo en esa dirección
        int enemy_piece_pointer_square = -1; //esto es la celda de la pieza enemiga en cuestión
        for (int n = 0; n < offset_limit; ++n){
            int target_square = enemy_king_target_square + offset_dir * (n + 1);
            int piece_on_target_square = flatten_board.index({target_square}).item<int>();
            bool is_target_enemy = (currentPlayer == 1 && piece_on_target_square < 0) || (currentPlayer == 0 && piece_on_target_square > 0); 
            bool is_target_ally = (currentPlayer == 1 && piece_on_target_square > 0) || (currentPlayer == 0 && piece_on_target_square < 0);

            if(is_target_enemy){
                //se suma el contador de piezas enemigas (alidas para el rey enemigo) en esa dirección
                if(n_straight_allies == 0){
                    n_straight_allies++;
                    enemy_piece_pointer_square = target_square;
                }else{
                    break;
                }
            }else if(is_target_ally){
                int abs_piece_ally = abs(piece_on_target_square);
                break;
            }
        }
    }
}


void ChessGameState::set_target_squares_attacked(){
    target_squares_attacked_by_enemy.clear();
    target_squares_on_check.clear();
    //squares_on_check_stuck.clear();
    on_check = false;
    count_checks_pieces = 0;

    std::vector<int> target_squares_attacked_by_enemy_vector;
    for(int start_square = 0; start_square < row_count*column_count; ++start_square){
        auto piece_idx = flatten_board.index({start_square}).item<int>();
        bool piece_is_ours = (currentPlayer == 1 && piece_idx > 0) || (currentPlayer == 0 && piece_idx < 0);
        if (piece_is_ours){
            auto squares = this->get_target_squares_piece(piece_idx, start_square);
            target_squares_attacked_by_enemy_vector.insert(target_squares_attacked_by_enemy_vector.end(), squares.begin(), squares.end());
        }
    }
    target_squares_attacked_by_enemy.insert(target_squares_attacked_by_enemy_vector.begin(), target_squares_attacked_by_enemy_vector.end());
    if(count_checks_pieces > 1){
        on_double_check = true;
    }
}


std::vector<int> ChessGameState::get_target_squares_piece(int piece_idx, int start_square){
    int abs_piece = abs(piece_idx);
    int row = start_square / row_count;
    int col = start_square % column_count;

    std::vector<int> target_squares;
    torch::Tensor valid_moves_fake = torch::zeros({8});

    if(sliding_pieces.find(piece_idx) != sliding_pieces.end()) {
        target_squares = this->set_valid_moves_sliding_piece(start_square, abs_piece, col, row, valid_moves_fake, false);
    }else if(abs_piece == 2){
        target_squares = this->set_valid_moves_knight_piece(start_square, col, row, valid_moves_fake, false);
    }else if(abs_piece == 5){
        target_squares = this->set_valid_moves_king_piece(start_square, col, row, valid_moves_fake, false);
    }else if(abs_piece == 6){
        target_squares = this->set_valid_moves_pawn_piece(start_square, col, row, valid_moves_fake, false);    
    }

    return target_squares;
}


std::pair<int, int> ChessGameState::set_next_state_king(int start_square, int direction, int piece_idx, bool set_board){
    std::pair<int, int> info_target;
    int direction_idx = direction % column_count;
    int n = direction / column_count;
    int offset_dir = sliding_direction_offsets[direction_idx];
    int target_square = start_square + offset_dir * (n + 1);

    int target_piece = flatten_board.index({target_square}).item<int>();
    if(set_board){
        flatten_board.index_put_({target_square}, piece_idx);
        if (n > 0){
            switch(target_square){
                case 2:
                    //enroque largo negro
                    flatten_board.index_put_({0}, 0);
                    flatten_board.index_put_({3}, -1);
                    castlingRigtsBlack.erase(std::remove(castlingRigtsBlack.begin(), castlingRigtsBlack.end(), 'q'), castlingRigtsBlack.end());
                    break;
                case 6:
                    //enroque corto negro
                    flatten_board.index_put_({7}, 0);
                    flatten_board.index_put_({5}, -1);
                    castlingRigtsBlack.erase(std::remove(castlingRigtsBlack.begin(), castlingRigtsBlack.end(), 'k'), castlingRigtsBlack.end());
                    break;
                case 58:
                    //enroque largo blanco
                    flatten_board.index_put_({56}, 0);
                    flatten_board.index_put_({59}, 1);
                    castlingRigtsWhite.erase(std::remove(castlingRigtsWhite.begin(), castlingRigtsWhite.end(), 'Q'), castlingRigtsWhite.end());
                    break;
                case 62:
                    //enroque corto blanco
                    flatten_board.index_put_({63}, 0);
                    flatten_board.index_put_({61}, 1);
                    castlingRigtsWhite.erase(std::remove(castlingRigtsWhite.begin(), castlingRigtsWhite.end(), 'K'), castlingRigtsWhite.end());
                    break;                   
            }
        }else{
            //caso de mover el rey una posición, eliminar los derechos de enroque
            if(currentPlayer == 1 && castlingRigtsWhite.length() > 0){
                castlingRigtsWhite.erase(std::remove(castlingRigtsWhite.begin(), castlingRigtsWhite.end(), 'K'), castlingRigtsWhite.end());
                castlingRigtsWhite.erase(std::remove(castlingRigtsWhite.begin(), castlingRigtsWhite.end(), 'Q'), castlingRigtsWhite.end());
            }else if(currentPlayer == 0 && castlingRigtsBlack.length() > 0){
                castlingRigtsBlack.erase(std::remove(castlingRigtsBlack.begin(), castlingRigtsBlack.end(), 'k'), castlingRigtsBlack.end());
                castlingRigtsBlack.erase(std::remove(castlingRigtsBlack.begin(), castlingRigtsBlack.end(), 'q'), castlingRigtsBlack.end());
            }
        }
    }
    info_target = std::make_pair(target_square, target_piece);

    return info_target;
}


torch::Tensor ChessGameState::simulate_next_state_pawn(int direction, int start_square, int row, int piece_idx, torch::Tensor& flatten_board_aux){
    const int subpromotions_limit = 64;
    auto dirs_vector = (currentPlayer == 1) ? directions_pawn_white : directions_pawn_black;
    std::set<int> dirs_vector_eat = {dirs_vector[0], dirs_vector[2]};
    bool is_eat_dir = dirs_vector_eat.find(direction) != dirs_vector_eat.end();
    if (direction >= subpromotions_limit){
        //caso de subpromociones
        int offset;
        int target_piece;
        switch(direction){
            case subpromotions_limit:
                //case subpromoción izquierda con alfil
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[7];
                    target_piece = 3;
                }else{
                    offset = sliding_direction_offsets[3];
                    target_piece = -3;
                }
                break;

            case subpromotions_limit+1:
                //case subpromoción izquierda con caballo
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[7];
                    target_piece = 2;
                }else{
                    offset = sliding_direction_offsets[3];
                    target_piece = -2;
                }
                break;

            case subpromotions_limit+2:
                //case subpromoción izquierda con torre
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[7];
                    target_piece = 1;
                }else{
                    offset = sliding_direction_offsets[3];
                    target_piece = -1;
                }
                break;   

            case subpromotions_limit+3:
                //case subpromoción recto con alfil
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[0];
                    target_piece = 3;
                }else{
                    offset = sliding_direction_offsets[4];
                    target_piece = -3;
                }
                break;   

            case subpromotions_limit+4:
                //case subpromoción recto con caballo
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[0];
                    target_piece = 2;
                }else{
                    offset = sliding_direction_offsets[4];
                    target_piece = -2;
                }
                break;  

            case subpromotions_limit+5:
                //case subpromoción recto con torre
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[0];
                    target_piece = 1;
                }else{
                    offset = sliding_direction_offsets[4];
                    target_piece = -1;
                }
                break;  

            case subpromotions_limit+6:
                //case subpromoción derecha con alfil
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[1];
                    target_piece = 3;
                }else{
                    offset = sliding_direction_offsets[5];
                    target_piece = -3;
                }
                break; 

            case subpromotions_limit+7:
                //case subpromoción derecha con caballo
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[1];
                    target_piece = 2;
                }else{
                    offset = sliding_direction_offsets[5];
                    target_piece = -2;
                }
                break; 

            case subpromotions_limit+8:
                //case subpromoción derecha con torre
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[1];
                    target_piece = 1;
                }else{
                    offset = sliding_direction_offsets[5];
                    target_piece = -1;
                }
                break; 

        }
        int target_square = start_square + offset;
        flatten_board_aux.index_put_({target_square}, target_piece);
    }else{
        int direction_idx = direction % column_count;
        int offset_dir = sliding_direction_offsets[direction_idx];
        int n = direction / column_count;
        int target_square = start_square + offset_dir * (n + 1);
        //caso de coronación
        bool finish_row = (row == 6 && currentPlayer == 0) || (row == 1 && currentPlayer == 1);
        if(finish_row){
            int target_piece;
            if(currentPlayer == 1){
                target_piece = 4;
            }else{
                target_piece = -4;
            }
            flatten_board_aux.index_put_({target_square}, target_piece);
        }else{
            //caso especial enPassant
            int target_piece = flatten_board.index({target_square}).item<int>();
            if(is_eat_dir && target_piece == 0){
                int target_en_passant = currentPlayer == 1 ? target_square + column_count : target_square - column_count;
                flatten_board_aux.index_put_({target_en_passant}, 0);
            }
            
            flatten_board_aux.index_put_({target_square}, piece_idx);
        }
    }
    return flatten_board_aux;
}


std::tuple<int, int, bool> ChessGameState::set_next_state_pawn(int direction, int start_square, int row, int piece_idx, bool set_board){
    std::tuple<int, int, bool> info_target;
    bool new_on_passant = false;
    const int subpromotions_limit = 64;
    auto dirs_vector = (currentPlayer == 1) ? directions_pawn_white : directions_pawn_black;
    std::set<int> dirs_vector_eat = {dirs_vector[0], dirs_vector[2]};
    bool is_eat_dir = dirs_vector_eat.find(direction) != dirs_vector_eat.end();
    if (direction >= subpromotions_limit){
        //caso de subpromociones
        int offset;
        int target_piece;
        switch(direction){
            case subpromotions_limit:
                //case subpromoción izquierda con alfil
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[7];
                    target_piece = 3;
                }else{
                    offset = sliding_direction_offsets[3];
                    target_piece = -3;
                }
                break;

            case subpromotions_limit+1:
                //case subpromoción izquierda con caballo
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[7];
                    target_piece = 2;
                }else{
                    offset = sliding_direction_offsets[3];
                    target_piece = -2;
                }
                break;

            case subpromotions_limit+2:
                //case subpromoción izquierda con torre
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[7];
                    target_piece = 1;
                }else{
                    offset = sliding_direction_offsets[3];
                    target_piece = -1;
                }
                break;   

            case subpromotions_limit+3:
                //case subpromoción recto con alfil
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[0];
                    target_piece = 3;
                }else{
                    offset = sliding_direction_offsets[4];
                    target_piece = -3;
                }
                break;   

            case subpromotions_limit+4:
                //case subpromoción recto con caballo
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[0];
                    target_piece = 2;
                }else{
                    offset = sliding_direction_offsets[4];
                    target_piece = -2;
                }
                break;  

            case subpromotions_limit+5:
                //case subpromoción recto con torre
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[0];
                    target_piece = 1;
                }else{
                    offset = sliding_direction_offsets[4];
                    target_piece = -1;
                }
                break;  

            case subpromotions_limit+6:
                //case subpromoción derecha con alfil
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[1];
                    target_piece = 3;
                }else{
                    offset = sliding_direction_offsets[5];
                    target_piece = -3;
                }
                break; 

            case subpromotions_limit+7:
                //case subpromoción derecha con caballo
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[1];
                    target_piece = 2;
                }else{
                    offset = sliding_direction_offsets[5];
                    target_piece = -2;
                }
                break; 

            case subpromotions_limit+8:
                //case subpromoción derecha con torre
                if (currentPlayer == 1){
                    offset = sliding_direction_offsets[1];
                    target_piece = 1;
                }else{
                    offset = sliding_direction_offsets[5];
                    target_piece = -1;
                }
                break; 

        }
        int target_square = start_square + offset;
        if(set_board){
            flatten_board.index_put_({target_square}, target_piece);
        }
        info_target = std::make_tuple(target_square, target_piece, new_on_passant);
    }else{
        int direction_idx = direction % column_count;
        int offset_dir = sliding_direction_offsets[direction_idx];
        int n = direction / column_count;
        int target_square = start_square + offset_dir * (n + 1);
        //caso de coronación
        bool finish_row = (row == 6 && currentPlayer == 0) || (row == 1 && currentPlayer == 1);
        if(finish_row){
            int target_piece;
            if(currentPlayer == 1){
                target_piece = 4;
            }else{
                target_piece = -4;
            }
            if(set_board){
                flatten_board.index_put_({target_square}, target_piece);
            }
            info_target = std::make_tuple(target_square, target_piece, new_on_passant);
        }else{
            if(set_board){
                //caso especial enPassant
                int target_piece = flatten_board.index({target_square}).item<int>();
                if(is_eat_dir && target_piece == 0){
                    int target_en_passant = currentPlayer == 1 ? target_square + column_count : target_square - column_count;
                    flatten_board.index_put_({target_en_passant}, 0);
                }
                
                flatten_board.index_put_({target_square}, piece_idx);
                //activar enPassant si se hace doble avance y hay peón enemigo adyacente
                if(n==1){
                    int col = target_square % column_count;
                    int right_piece = 0;
                    int left_piece = 0;
                    int enemy_pawn = currentPlayer == 1 ? -6 : 6;
                    int target_en_passant = currentPlayer == 1 ? target_square + column_count : target_square - column_count;
                    if(col == 0){
                        right_piece = flatten_board.index({target_square + 1}).item<int>();
                    }else if(col == column_count - 1){
                        left_piece = flatten_board.index({target_square - 1}).item<int>();
                    }else{
                        right_piece = flatten_board.index({target_square + 1}).item<int>();
                        left_piece = flatten_board.index({target_square - 1}).item<int>();
                    }
                    if(right_piece == enemy_pawn || left_piece == enemy_pawn){
                        enPassantTarget = target_en_passant;
                        new_on_passant = true;
                    }
                }
            }
            int target_piece = flatten_board.index({target_square}).item<int>();
            info_target = std::make_tuple(target_square, target_piece, new_on_passant);
        }
    }
    return info_target;
}


void ChessGameState::reset(){
    history_m.clear();
    repetitions_per_position.clear();

    on_check = false;
    on_double_check = false;
    currentPlayer = 1;
    castlingRigtsWhite = "KQ";
    castlingRigtsBlack = "kq";
    enPassantTarget = -1;
    halfMoveClock = 0;
    fullMoveNumber = 1;
    cod_fem = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    this->fentoBoard();

    int time_channels = m_features*t_timesteps;

    torch::Tensor empty_board_m = torch::zeros({8,8,3});

    for (int i=0; i<t_timesteps-1;++i){
        history_m.push_back(empty_board_m);
    }

    std::vector<torch::Tensor> first_board_m;
    torch::Tensor empty_repetitions_m = torch::zeros({8,8,2});
    first_board_m.push_back(empty_repetitions_m);

    first_board_m.push_back(board.unsqueeze(-1).clone());
    auto first_state_m = torch::concat(first_board_m, 2);
    history_m.push_back(first_state_m);
}

void ChessGameState::fentoBoard(){
    board = torch::zeros({row_count, column_count}).to(torch::kInt16);
    int row = 0, col = 0;
    //se parte el fen por el espacio para procesarlo por partes
    std::istringstream iss(cod_fem);
    std::string fem_part;

    int counter_parts = 0;
    while (iss >> fem_part) {
        if (counter_parts == 0){
            for (char c : fem_part) {
                if (isdigit(c)) {
                    col += c - '0'; // Aumenta la columna por el número de espacios vacíos
                } else if (c == '/') {
                    row++; // Mueve a la siguiente fila
                    col = 0;
                } else {
                    int idx_piece = pieceToNumber[c];
                    board.index_put_({row, col}, idx_piece); // Asigna la pieza al tablero
                    col++;
                }
            }
        }else if (counter_parts == 1){
            if (fem_part == "w"){
                currentPlayer = 1;
            }else{
                currentPlayer = 0;
            }
        }else if (counter_parts == 2){
            //analisis de enroques
            castlingRigtsBlack = "";
            castlingRigtsWhite = "";
            for(char c: fem_part){
                if (std::isupper(c)) {
                    castlingRigtsWhite += c;
                } else if (std::islower(c)) {
                    castlingRigtsBlack += c;
                }
            }
        }else if (counter_parts == 3){
            if(fem_part == "-"){
                enPassantTarget = -1;
            }else{
                auto col_str = fem_part[0];
                auto row_str = fem_part[1];
                int col_on_passant = colCharToColNum[col_str];
                int row_from_down = row_str - '0';
                int row_on_passant = row_count - row_from_down;

                enPassantTarget = row_count*row_on_passant + col_on_passant;
            }
            break;
        }
        counter_parts+=1;
    }

    flatten_board = board.view({row_count*column_count});
    //std::cout << flatten_board.sizes() << std::endl;
}

void ChessGameState::boardtoFEN(){
    cod_fem = "";
    std::string castling_rights_encoded;
    std::string en_passant_encoded;

    for (int row = 0; row < row_count; ++row) {
        if (row > 0) {
            cod_fem += "/";
        }
        int emptySquares = 0;
        for (int col = 0; col < column_count; ++col) {
            int piece = board.index({row, col}).item<int>();
            if (piece == 0) { // Suponiendo que 0 representa un cuadro vacío
                ++emptySquares;
            } else {
                if (emptySquares > 0) {
                    cod_fem += std::to_string(emptySquares);
                    emptySquares = 0;
                }
                auto it = numberToPiece.find(piece);
                cod_fem += it->second;
            }
        }
        if (emptySquares > 0) {
            cod_fem += std::to_string(emptySquares);
        }
    }
    //codificando jugador que mueve
    if (currentPlayer == 1){
        cod_fem += " w ";
    }else{
        cod_fem += " b ";
    }

    //codificando estado del enroque
    if (castlingRigtsWhite == "" && castlingRigtsBlack == ""){
        cod_fem += "- ";
    }else{
        cod_fem += castlingRigtsWhite + castlingRigtsBlack + " ";
    }

    //codificando en passant
    if (enPassantTarget == -1){
        cod_fem += "- ";
    }else{
        auto decoded_en_passant = int_square_to_string(enPassantTarget);
        cod_fem += decoded_en_passant + " ";
    }

    //añadiendo numero de jugadas sin comer peones
    cod_fem += std::to_string(halfMoveClock) + " ";

    //añadiendo numero de jugadas de partida
    cod_fem += std::to_string(fullMoveNumber) + " ";
}

torch::Tensor ChessGameState::get_encoded_state(){
    // los planos M (de tablero) tienen que mostrarse según la perspectiva del jugador actual
    // la idea es que history m tenga los tableros en crudo y luego last_t_m los tenga decodificados
    // tengrá entonces history_m para cada jugada 3 planos (dos indicando el número de repeticiones) y otro el board como tal

    std::vector<torch::Tensor> last_t_m;
    for(const torch::Tensor &m: history_m){
        //std::cout << m.sizes() << std::endl;
        auto repetitions_planes = m.index({"...", torch::indexing::Slice(0,2)});
        last_t_m.push_back(repetitions_planes);
        auto board_raw_plane = m.index({"...", 2});
        //std::cout << board_raw_plane.sizes() << std::endl;
        auto order_m_planes = (currentPlayer == 1) ? order_m_planes_white:order_m_planes_black;
        for(const int idx_piece: order_m_planes){
            auto mask = (board_raw_plane == idx_piece).to(torch::kInt16).unsqueeze(-1);
            last_t_m.push_back(mask);
        }
    }
    //formamos el tensor actial con el resto de variables
    torch::Tensor constant = torch::ones({8,8,1});
    last_t_m.push_back(constant * currentPlayer);
    //el plano lo normalizamos suponiendo que el máximo de jugadas es 300
    last_t_m.push_back(constant * fullMoveNumber / 300.0);

    auto castle_rights_plane = torch::zeros({8,8,4});
    //el plano de enroque también depende del jugador actual
    std::string castlings_rights_total = castlingRigtsWhite + castlingRigtsBlack;
    if (castlings_rights_total.length() > 0){
        int idx_perspective;
        for (char c: castlings_rights_total){
            if (c == 'K'){
                idx_perspective = (currentPlayer == 1) ? 0 : 2;
                castle_rights_plane.index_put_({"...", idx_perspective}, 1);
            }else if (c == 'Q'){
                idx_perspective = (currentPlayer == 1) ? 1 : 3;
                castle_rights_plane.index_put_({"...", idx_perspective}, 1);
            }else if (c == 'k'){
                idx_perspective = (currentPlayer == 0) ? 0 : 2;
                castle_rights_plane.index_put_({"...", idx_perspective}, 1);
            }else if (c == 'q'){
                idx_perspective = (currentPlayer == 0) ? 1 : 3;
                castle_rights_plane.index_put_({"...", idx_perspective}, 1);
            }
        }
    }
    last_t_m.push_back(castle_rights_plane);
    //plano que cuenta el número de jugadas sin comer peones
    last_t_m.push_back(constant * halfMoveClock / static_cast<float>(limit_plays_draw));

    //state_encoded es el tensor (8,8, 119);
    auto state_encoded = torch::concat(last_t_m, 2);

    //el state tiene que ser (119, 8, 8) en forma imágenes de torch
    state_encoded = state_encoded.permute({2, 0, 1}).unsqueeze(0);

    return state_encoded;
}

torch::Tensor ChessGameState::get_encoded_state_vm(){
    std::vector<torch::Tensor> last_t_m;

    auto m = history_m[t_timesteps-1];
    auto board_raw_plane = m.index({"...", 2});
    auto order_m_planes = (currentPlayer == 1) ? order_m_planes_white:order_m_planes_black;
    for(const int idx_piece: order_m_planes){
        auto mask = (board_raw_plane == idx_piece).to(torch::kInt16).unsqueeze(-1);
        last_t_m.push_back(mask);
    }

    //formamos el tensor actial con el resto de variables
    torch::Tensor constant = torch::ones({8,8,1});
    last_t_m.push_back(constant * currentPlayer);

    auto castle_rights_plane = torch::zeros({8,8,4});
    //el plano de enroque también depende del jugador actual
    std::string castlings_rights_total = (currentPlayer == 1) ? castlingRigtsWhite + castlingRigtsBlack : castlingRigtsBlack + castlingRigtsWhite;
    if (castlings_rights_total.length() > 0){
        for (char c: castlings_rights_total){
            if (c == 'K'){
                castle_rights_plane.index_put_({"...", 0}, 1);
            }else if (c == 'Q'){
                castle_rights_plane.index_put_({"...", 1}, 1);
            }else if (c == 'k'){
                castle_rights_plane.index_put_({"...", 2}, 1);
            }else if (c == 'q'){
                castle_rights_plane.index_put_({"...", 3}, 1);
            }
        }
    }
    last_t_m.push_back(castle_rights_plane);

    auto en_passant_plane = torch::zeros({8,8,1});
    if(enPassantTarget != -1){
        int row = enPassantTarget / row_count;
        int col = enPassantTarget % column_count;
        en_passant_plane.index_put_({row, col, 0}, 1);
    }

    last_t_m.push_back(en_passant_plane);

    //state_encoded es el tensor (8,8, 18);
    auto state_encoded = torch::concat(last_t_m, 2);

    //el state tiene que ser (18, 8, 8) en forma imágenes de torch
    state_encoded = state_encoded.permute({2, 0, 1}).unsqueeze(0);

    return state_encoded;
}

torch::Tensor ChessGameState::get_valid_moves(){
    torch::Tensor valid_moves_tensor = torch::zeros({row_count,column_count,73});
    //creo que la idea es recorrer el tablero 8*8 y según la pieza que sea, ver sus posibles movimientos
    for(int start_square = 0; start_square < row_count*column_count; ++start_square){
        auto piece_idx = flatten_board.index({start_square}).item<int>();
        //comprobamos que la pieza sea nuestra y que no esté en el listado de piezas clavadas en jaque
        bool piece_is_ours = (currentPlayer == 1 && piece_idx > 0) || (currentPlayer == 0 && piece_idx < 0);
        if(piece_is_ours){
            this->set_valid_moves_piece(piece_idx, start_square, valid_moves_tensor);
        }
    }
    return valid_moves_tensor;
}


std::vector<int> ChessGameState::set_valid_moves_sliding_piece(int start_square, int abs_piece, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves){
    std::vector<int> target_squares;
    for(int direction_idx=0; direction_idx < 8; ++direction_idx){
        //hay que comprobar si la pieza es reina, torre o alfil para comprobar las direcciones o pasar a la siguiente
        bool is_even = direction_idx % 2 == 0;
        //si se trata de torre y la direción es impar (diagonal), no se comprueba; lo contrario con alfiles
        if (abs_piece == 1 && !is_even){
            continue;
        }else if(abs_piece == 3 && is_even){
            continue;
        }

        int offset_dir = sliding_direction_offsets[direction_idx]; //offset de squares por celda
        int offset_limit = sliding_pieces_to_edge_dict[start_square][direction_idx]; //numero de celdas totales posibles en esa dirección
        for (int n = 0; n < offset_limit; ++n){
            int target_square = start_square + offset_dir * (n + 1);
            int piece_on_target_square = flatten_board.index({target_square}).item<int>();
            bool is_target_enemy = (currentPlayer == 1 && piece_on_target_square < 0) || (currentPlayer == 0 && piece_on_target_square > 0); 
            bool is_target_ally = (currentPlayer == 1 && piece_on_target_square > 0) || (currentPlayer == 0 && piece_on_target_square < 0); 

            //los target_squares tienen que incluir piezas aliadas o sin celda ya que son celdas donde el rey enemido no tiene permitido moverse o capurar
            if (!only_valid_moves && (piece_on_target_square == 0 || is_target_ally)){
                target_squares.push_back(target_square);
            }

            //si es pieza propia se para en esa dirección
            if (is_target_ally){
                break;
            }

            //si pasa ese bloque entonces se considera jugada válida y se actualiza el tensor de valid_moves
            //codificamos el indice del movimiento suponiendo que el offset n son las filas y direction_idx son las columnas

            //hay que comprobar el caso que sea jaque para añadir la jugada valida

            //se hace así para evitar la búsqueda en la lista si no hay jaques
            if (only_valid_moves){
                int idx_move = column_count*n + direction_idx;
                auto valid_move_check = this->on_check_control_move(target_square);
                std::vector<int> action_idx = {row, col, idx_move};
                auto valid_move_check_after = check_validation_move(action_idx);
                if (valid_move_check && valid_move_check_after){
                    valid_moves_tensor.index_put_({row, col, idx_move}, 1);
                }
            }

            //si hay pieza enemiga se toma como movimiento válido (comer) y se corta ahí
            if (is_target_enemy){
                if(!only_valid_moves && abs(piece_on_target_square) == 5){
                    on_check = true;
                    count_checks_pieces++;
                    target_squares_on_check.insert(start_square);
                    //se aplica el deslizamiento de forma inversa para añadir los target_squares_on_check
                    if(n > 0){
                        //std::cout << "Deslizando por la diagonal de jaque" << std::endl;
                        for(int n_i = n-1; n_i >= 0; --n_i){
                            int target_square = start_square + offset_dir * (n_i + 1);
                            target_squares_on_check.insert(target_square);
                        }
                    }
                }else{
                    break;
                }
            }
        }
    }

    return target_squares;
}


bool ChessGameState::on_check_control_move(int target_square){
    bool valid_move = false; 
    
    if(!on_check){
        //si no estamos en jaque, el movimiento es válido automáticamente (recordamos que aquí no se pasan las piezas clavadas)
        valid_move = true;
    }else{
        auto it = std::find(target_squares_on_check.begin(), target_squares_on_check.end(), target_square);
        bool target_square_on_check_range = it != target_squares_on_check.end();
        if(target_square_on_check_range){
            valid_move = true;
        }
    }

    return valid_move;
}

bool ChessGameState::check_validation_move(const std::vector<int>& action_idx){
    int row = action_idx[0];
    int col = action_idx[1];
    int direction = action_idx[2];

    //sabemos que celda = row*8 + col
    int start_square = row*row_count + col;
    int piece_idx = flatten_board.index({start_square}).item<int>();
    int abs_piece = abs(piece_idx);

    //se simula la acción en el tablero
    auto flatten_board_aux = flatten_board.clone();
    flatten_board_aux.index_put_({start_square}, 0);
    int target_piece;

    if(sliding_pieces.find(piece_idx) != sliding_pieces.end()) {
        int n = direction / column_count;

        int direction_idx = direction % column_count;
        //formulas para despejar la ecuación x = 8*n + direction
        int offset_dir = sliding_direction_offsets[direction_idx];
        int target_square = start_square + offset_dir * (n + 1);

        target_piece = flatten_board_aux.index({target_square}).item<int>();

        flatten_board_aux.index_put_({target_square}, piece_idx);
    //tratamiento de caballos
    }else if(abs_piece == 2){
        int direction_idx = direction - 56;
        int target_square = start_square + knight_direction_offsets[direction_idx];

        target_piece = flatten_board_aux.index({target_square}).item<int>();

        flatten_board_aux.index_put_({target_square}, piece_idx);        

    }else if(abs_piece == 6){
        flatten_board_aux = this->simulate_next_state_pawn(direction, start_square, row, piece_idx, flatten_board_aux);
    }

    int king_square;
    if (currentPlayer == 1){
        king_square = torch::nonzero(flatten_board_aux == 5).item<int>();
    }else{
        king_square = torch::nonzero(flatten_board_aux == -5).item<int>();
    }
    bool on_check_new = valid_king_on_check(flatten_board_aux, king_square);
    return !on_check_new;
}


bool ChessGameState::valid_king_on_check(const torch::Tensor& flatten_board_aux, int king_square){
    bool on_check_new = false; 
    for(int direction_idx=0; direction_idx < 8; ++direction_idx){
        bool is_even = direction_idx % 2 == 0;
        int offset_dir = sliding_direction_offsets[direction_idx]; //offset de squares por celda
        int offset_limit = sliding_pieces_to_edge_dict[king_square][direction_idx]; //numero de celdas totales posibles en esa dirección
        for (int n = 0; n < offset_limit; ++n){
            int target_square = king_square + offset_dir * (n + 1);
            int piece_on_target_square = flatten_board_aux.index({target_square}).item<int>();
            bool is_target_enemy = (currentPlayer == 1 && piece_on_target_square < 0) || (currentPlayer == 0 && piece_on_target_square > 0); 
            bool is_target_ally = (currentPlayer == 1 && piece_on_target_square > 0) || (currentPlayer == 0 && piece_on_target_square < 0); 
            int abs_target_piece = abs(piece_on_target_square);
            if (is_target_ally){
                break;
            }

            if (is_target_enemy){
                //caso reina enemiga esté apuntando al rey
                if(abs_target_piece == 4){
                    on_check_new = true;
                }
                //caso de que la dirección sea cruz y la torre enemiga esté apuntando al rey
                else if(abs_target_piece == 1 && is_even){
                    on_check_new = true;
                //caso alfil enemigo esté apuntando al rey
                }else if(abs_target_piece == 3 && !is_even){
                    on_check_new = true;
                }else{
                    break;
                }
            }
        }
    }

    return on_check_new;
}


std::string ChessGameState::decode_action(const torch::Tensor& action_idx){

    std::string decoded_action;
    //no es la decodificación perfecta porque no hay jaque
    int row = action_idx.index({0}).item<int>();
    int col = action_idx.index({1}).item<int>();
    int direction = action_idx.index({2}).item<int>();

    //sabemos que celda = row*8 + col
    int start_square = row*row_count + col;
    int piece_idx = flatten_board.index({start_square}).item<int>();
    int abs_piece = abs(piece_idx);
    int target_piece;
    int target_square;

    if(sliding_pieces.find(piece_idx) != sliding_pieces.end()) {
        int n = direction / column_count;

        int direction_idx = direction % column_count;
        //formulas para despejar la ecuación x = 7*n + direction
        int offset_dir = sliding_direction_offsets[direction_idx];
        target_square = start_square + offset_dir * (n + 1);

        auto char_piece = numberToPieceDecode[abs_piece];
        auto decoded_target_square = int_square_to_string(target_square);
        auto target_piece = flatten_board.index({target_square}).item<int>();

        decoded_action = std::string(1, char_piece);

        if(target_piece != 0){
            decoded_action += std::string(1, 'x');
        }

        decoded_action += decoded_target_square;

    }else if(abs_piece == 2){
        int direction_idx = direction - 56;
        target_square = start_square + knight_direction_offsets[direction_idx];
        auto char_piece = numberToPieceDecode[abs_piece];
        auto decoded_target_square = int_square_to_string(target_square);
        auto target_piece = flatten_board.index({target_square}).item<int>();

        decoded_action = std::string(1, char_piece);

        if(target_piece != 0){
            decoded_action += std::string(1, 'x');
        }

        decoded_action += decoded_target_square;

    //tratamiento de peones, incluida la coronación 
    }else if(abs_piece == 6){
        auto info_target = this->set_next_state_pawn(direction, start_square, row, piece_idx, false);
        target_square = std::get<0>(info_target);
        target_piece = std::get<1>(info_target);
        auto decoded_target_square = int_square_to_string(target_square);
        //std::cout << "target piece: " << target_piece << std::endl;

        bool finish_row = (row == 6 && currentPlayer == 0) || (row == 1 && currentPlayer == 1);
        if (finish_row){
            auto target_piece_decoded = numberToPieceDecode[abs(target_piece)];
            decoded_action = std::string(1, target_piece_decoded);

            if(target_piece != 0){
                decoded_action += std::string(1, 'x');
            }

            decoded_action += decoded_target_square;
        }else{
            if(target_piece != 0){
                int col = start_square % column_count;
                auto col_char = colNumToColChar[col];

                decoded_action = std::string(1, col_char);

                if(target_piece != 0){
                    decoded_action += std::string(1, 'x');
                }

                decoded_action += decoded_target_square;
            }else{
                decoded_action = decoded_target_square;
            }

        }
    //tratamiento de rey, incluido el enroque
    }else if (abs_piece == 5){
        auto info_target = this->set_next_state_king(start_square, direction, piece_idx, false);
        target_square = info_target.first;
        target_piece = info_target.second;
        auto char_piece = numberToPieceDecode[abs_piece];
        auto decoded_target_square = int_square_to_string(target_square);

        decoded_action = std::string(1, char_piece);
        
        if(target_piece != 0){
            decoded_action += std::string(1, 'x');
        }

        decoded_action += decoded_target_square;
    }
    return decoded_action;
}

std::string ChessGameState::int_square_to_string(int target_square){
    int row = row_count - (target_square / row_count);
    int col = target_square % column_count;
    auto col_char = colNumToColChar[col];
    std::string decoded_target_square = col_char + std::to_string(row);

    return decoded_target_square;
}

std::vector<int> ChessGameState::set_valid_moves_knight_piece(int start_square, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves){
    std::vector<int> target_squares;
    for(int direction_idx=0; direction_idx < 8; ++direction_idx){
        bool offset_allowed = knight_pieces_to_edge_dict[start_square][direction_idx];
        //se hace un check para ver si se permite el movimiento
        if (offset_allowed){
            int target_square = start_square + knight_direction_offsets[direction_idx];
            int piece_on_target_square = flatten_board.index({target_square}).item<int>();

            //solo si la pieza no es del jugador se permite la jugada o si no hay ninguna
            bool target_piece_enemy = (currentPlayer == 1 && piece_on_target_square < 0) || (currentPlayer == 0 && piece_on_target_square > 0);
            bool is_target_ally = (currentPlayer == 1 && piece_on_target_square > 0) || (currentPlayer == 0 && piece_on_target_square < 0); 
            bool empty_target_square = piece_on_target_square == 0;
            if ((target_piece_enemy || empty_target_square) && only_valid_moves){
                //control de jaque
                auto valid_move_check = this->on_check_control_move(target_square);
                int idx_move = 56 + direction_idx;
                std::vector<int> action_idx = {row, col, idx_move};
                auto valid_move_check_after = check_validation_move(action_idx);
                if(valid_move_check && valid_move_check_after){
                    //std::cout << "Habilitando jugada de salto de caballo" << std::endl;
                    valid_moves_tensor.index_put_({row, col, idx_move}, 1);
                }
            }else if (!only_valid_moves){
                if(empty_target_square || is_target_ally){
                    target_squares.push_back(target_square);
                }else if(target_piece_enemy && abs(piece_on_target_square) == 5){
                    //caso de jaque
                    on_check = true;
                    count_checks_pieces++;
                    target_squares_on_check.insert(start_square);
                }
            }
        }
    }
    return target_squares;
}

std::vector<int> ChessGameState::set_valid_moves_pawn_piece(int start_square, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves){
    std::vector<int> target_squares;
    auto dirs_vector = (currentPlayer == 1) ? directions_pawn_white : directions_pawn_black;
    std::set<int> dirs_vector_eat = {dirs_vector[0], dirs_vector[2]};
    bool finish_row = (row == 6 && currentPlayer == 0) || (row == 1 && currentPlayer == 1);
    for(int direction_idx: dirs_vector){
        //check de si el movimiento está permitido
        bool is_eat_dir = dirs_vector_eat.find(direction_idx) != dirs_vector_eat.end();
        if (pawn_pieces_to_edge_dict[start_square][direction_idx]){

            int offset = sliding_direction_offsets[direction_idx];

            int target_square = start_square + offset;
            int piece_on_target_square = flatten_board.index({target_square}).item<int>();
            bool is_target_ally = (currentPlayer == 1 && piece_on_target_square > 0) || (currentPlayer == 0 && piece_on_target_square < 0); 
            bool target_piece_enemy = (currentPlayer == 1 && piece_on_target_square < 0) || (currentPlayer == 0 && piece_on_target_square > 0);
            auto valid_move_check = this->on_check_control_move(target_square);
            int idx_move = direction_idx;
            //casos a tratar:
            if (only_valid_moves){
                if (piece_on_target_square == 0){
                    //caso movimientos hacia adelante
                    if(direction_idx == 0 || direction_idx == 4){
                        if(valid_move_check){
                            if(finish_row){
                                //indices de subpromoción se les suma tres por que estamos en la dirección de jugar recto
                                auto subpromotions_vector_idx = idx_subpromotions + 3;
                                int idx_move1 = subpromotions_vector_idx.index({0}).item<int>();
                                int idx_move2 = subpromotions_vector_idx.index({1}).item<int>();
                                int idx_move3 = subpromotions_vector_idx.index({2}).item<int>();
                                std::vector<int> action_idx1 = {row, col, idx_move1};
                                auto valid_move_check_after1 = check_validation_move(action_idx1);
                                std::vector<int> action_idx2 = {row, col, idx_move2};
                                auto valid_move_check_after2 = check_validation_move(action_idx2);
                                std::vector<int> action_idx3 = {row, col, idx_move3};
                                auto valid_move_check_after3 = check_validation_move(action_idx3);
                                bool valid_move = (valid_move_check_after1 == 1 && valid_move_check_after2 == 1 && valid_move_check_after3 == 1);
                                if(valid_move){
                                    valid_moves_tensor.index_put_({row, col, subpromotions_vector_idx}, 1);
                                }
                            }
                            std::vector<int> action_idx = {row, col, idx_move};
                            auto valid_move_check_after = check_validation_move(action_idx);
                            if(valid_move_check_after){
                                valid_moves_tensor.index_put_({row, col, idx_move}, 1);
                            }
                            
                        }
                        //caso para dos saltos
                        if ((row == 1 && currentPlayer == 0) || (row == 6 && currentPlayer == 1)){
                            target_square += offset;
                            piece_on_target_square = flatten_board.index({target_square}).item<int>();
                            valid_move_check = this->on_check_control_move(target_square);
                            if (piece_on_target_square == 0 && valid_move_check){
                                std::vector<int> action_idx = {row, col, idx_move + column_count};
                                auto valid_move_check_after = check_validation_move(action_idx);
                                //se suma 7 al anterior por la formula idx_move = 7*(offsets-1) + dir_idx
                                if(valid_move_check_after){
                                    idx_move += column_count;
                                    //std::cout << "Habilitando jugada de dos daltos de peón" << std::endl;
                                    valid_moves_tensor.index_put_({row, col, idx_move}, 1);
                                }
                            }
                        }
                    }else if(is_eat_dir && enPassantTarget == target_square){
                        valid_move_check = this->on_check_control_move(enPassantTarget);
                        std::vector<int> action_idx = {row, col, direction_idx};
                        auto valid_move_check_after = check_validation_move(action_idx);
                        if(valid_move_check && valid_move_check_after){
                            valid_moves_tensor.index_put_({row, col, direction_idx}, 1);
                        }
                    }
                }//casos de comer pieza enemiga (sin incluir al paso)
                else if(is_eat_dir){
                    if (target_piece_enemy && only_valid_moves && valid_move_check){
                        //se pone como validos el caso de comer incluyendo coronando reina
                        std::vector<int> action_idx = {row, col, idx_move};
                        auto valid_move_check_after = check_validation_move(action_idx);
                        if(valid_move_check_after){
                            valid_moves_tensor.index_put_({row, col, idx_move}, 1);
                        }
                        //caso de subpromoción
                        if (finish_row){
                            //caso de subpromoción comiendo a la izquierda
                            if (direction_idx == 7 || direction_idx == 3){
                                int idx_move1 = idx_subpromotions.index({0}).item<int>();
                                int idx_move2 = idx_subpromotions.index({1}).item<int>();
                                int idx_move3 = idx_subpromotions.index({2}).item<int>();
                                std::vector<int> action_idx1 = {row, col, idx_move1};
                                auto valid_move_check_after1 = check_validation_move(action_idx1);
                                std::vector<int> action_idx2 = {row, col, idx_move2};
                                auto valid_move_check_after2 = check_validation_move(action_idx2);
                                std::vector<int> action_idx3 = {row, col, idx_move3};
                                auto valid_move_check_after3 = check_validation_move(action_idx3);
                                bool valid_move = (valid_move_check_after1 == 1 && valid_move_check_after2 == 1 && valid_move_check_after3 == 1);
                                if(valid_move){
                                    valid_moves_tensor.index_put_({row, col, idx_subpromotions}, 1);
                                }
                            //caso de subpromoción comiendo a la derecha
                            }else{
                                auto subpromotions_vector_idx = idx_subpromotions + 6;
                                int idx_move1 = subpromotions_vector_idx.index({0}).item<int>();
                                int idx_move2 = subpromotions_vector_idx.index({1}).item<int>();
                                int idx_move3 = subpromotions_vector_idx.index({2}).item<int>();
                                std::vector<int> action_idx1 = {row, col, idx_move1};
                                auto valid_move_check_after1 = check_validation_move(action_idx1);
                                std::vector<int> action_idx2 = {row, col, idx_move2};
                                auto valid_move_check_after2 = check_validation_move(action_idx2);
                                std::vector<int> action_idx3 = {row, col, idx_move3};
                                auto valid_move_check_after3 = check_validation_move(action_idx3);
                                bool valid_move = (valid_move_check_after1 == 1 && valid_move_check_after2 == 1 && valid_move_check_after3 == 1);
                                if(valid_move){
                                    valid_moves_tensor.index_put_({row, col, idx_subpromotions+6}, 1);
                                }
                            }
                        }
                    }
                }
            }else if(!only_valid_moves && is_eat_dir){
                if(piece_on_target_square == 0 || is_target_ally){
                    target_squares.push_back(target_square);
                }else if(target_piece_enemy && abs(piece_on_target_square) == 5){
                    on_check = true;
                    count_checks_pieces++;
                    target_squares_on_check.insert(start_square);
                }
            }
        }
    }
    return target_squares;
}

void ChessGameState::set_valid_moves_castling(int direction_idx, int col, int row, int idx_move_target, torch::Tensor& valid_moves_tensor, std::vector<int>& target_squares_castling){
    int n_offset = target_squares_castling.size();
    //empieza en el offset uno ya que el primero ha sido comprobado previamente
    for(int n = 0; n<n_offset; ++n){
        int target_square_castling = target_squares_castling[n];
        int piece_on_target_square = flatten_board.index({target_square_castling}).item<int>();
        auto it = std::find(target_squares_attacked_by_enemy.begin(), target_squares_attacked_by_enemy.end(), target_square_castling);
        if(piece_on_target_square != 0 || it != target_squares_attacked_by_enemy.end()){
            //se corta y sale en el caso que haya pieza en medio o enemigo ataque una de esas celdas
            break;
        //si se llegan a comprobar todos los pasos de enroque, entonces se puede enrocar
        }else if(n == (n_offset - 1)){
            //falta añadir que no sean fichas controladas por piezas enemigas
            valid_moves_tensor.index_put_({row, col, idx_move_target}, 1);
        }
    }                  
}


std::vector<int> ChessGameState::set_valid_moves_king_piece(int start_square, int col, int row, torch::Tensor& valid_moves_tensor, bool only_valid_moves){
    std::vector<int> target_squares;
    for(int direction_idx=0; direction_idx < 8; ++direction_idx){
        //check de si el movimiento está permitido
        if (king_piece_to_edge_dict[start_square][direction_idx]){
            int offset = sliding_direction_offsets[direction_idx];
            int target_square = start_square + offset;
            int piece_on_target_square = flatten_board.index({target_square}).item<int>();

            bool square_target_enemy_piece = (currentPlayer == 1 && piece_on_target_square < 0) || (currentPlayer == 0 && piece_on_target_square > 0);
            bool is_target_ally = (currentPlayer == 1 && piece_on_target_square > 0) || (currentPlayer == 0 && piece_on_target_square < 0); 
            //caso movimiento normal del rey
            auto it = std::find(target_squares_attacked_by_enemy.begin(), target_squares_attacked_by_enemy.end(), target_square);
            bool target_square_attacked_by_enemy = it != target_squares_attacked_by_enemy.end();
            //se comprueba que no haya pieza o que haya enemiga y que además no sea una celda atacada por el rival
            bool no_pieces_target_square = piece_on_target_square == 0;
            if ((no_pieces_target_square || square_target_enemy_piece) && !target_square_attacked_by_enemy && only_valid_moves){
                int idx_move = direction_idx;
                valid_moves_tensor.index_put_({row, col, idx_move}, 1);
                //casos de enroque
                //aunque quede feo con muchos ifs es mejor así por eficiencia ya que antes so ponen los ifs más probables para evitar
                //comprobaciones innecesarias
                if (no_pieces_target_square && !on_check){
                    if (currentPlayer == 1 && castlingRigtsWhite.length() > 0){
                        //check de enroque blanco
                        if (direction_idx == 2 && castlingRigtsWhite.find('K') != std::string::npos) {
                            this->set_valid_moves_castling(direction_idx, col, row, idx_move+column_count, valid_moves_tensor, target_squares_white_castling_kside);                        
                        }else if(direction_idx == 6 && castlingRigtsWhite.find('Q') != std::string::npos){
                            this->set_valid_moves_castling(direction_idx, col, row, idx_move+column_count, valid_moves_tensor, target_squares_white_castling_qside);
                        }
                    //check de enroque negro
                    }else if(currentPlayer == 0 && castlingRigtsBlack.length() > 0){
                        if (direction_idx == 2 && castlingRigtsBlack.find('k') != std::string::npos) {
                            this->set_valid_moves_castling(direction_idx, col, row, idx_move+column_count, valid_moves_tensor, target_squares_black_castling_kside);
                        }else if(direction_idx == 6 && castlingRigtsBlack.find('q') != std::string::npos){
                            this->set_valid_moves_castling(direction_idx, col, row, idx_move+column_count, valid_moves_tensor, target_squares_black_castling_qside);
                        }
                    }
                }
            }else if(!only_valid_moves && (no_pieces_target_square || is_target_ally)){
                target_squares.push_back(target_square);
            }
        }
    }
    return target_squares;
}


void ChessGameState::set_valid_moves_piece(int piece_idx, int start_square, torch::Tensor& valid_moves_tensor){
    int abs_piece = abs(piece_idx);
    //tratamiento de sliding pieces (reina, torre, alfil)
    int row = start_square / row_count;
    int col = start_square % column_count;

    if (abs_piece == 5){
        //tratamiento del rey incluyendo enroque. Se pueden usar los diccionarios de sliding_pieces pero se pone en otro metodo por temas de enroque y que solo se mueve un paso
        auto target_squares = this->set_valid_moves_king_piece(start_square, col, row, valid_moves_tensor, true);
    }
    //estos solo se miran si no estamos en doble check
    if (!on_double_check){
        //lo primero es comprobar el jaque si se quita del tablero la pieza a comprobar
        if(sliding_pieces.find(piece_idx) != sliding_pieces.end()) {
            auto target_squares = this->set_valid_moves_sliding_piece(start_square, abs_piece, col, row, valid_moves_tensor, true);
        //tratamiento de caballos
        }else if(abs_piece == 2){
            auto target_squares = this->set_valid_moves_knight_piece(start_square, col, row, valid_moves_tensor, true);
        //tratamiento de peones, incluida la coronación 
        }else if(abs_piece == 6){
            auto target_squares = this->set_valid_moves_pawn_piece(start_square, col, row, valid_moves_tensor, true);
        //tratamiento del rey incluyendo enroque. Se pueden usar los diccionarios de sliding_pieces pero se pone en otro metodo por temas de enroque y que solo se mueve un paso
        }
    }
}


float ChessGameState::get_opponent_value(float value){
    return -value;
}