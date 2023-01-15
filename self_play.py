import sys
import time
from copy import deepcopy

import engine

sys.path.append(sys.path[0] + "/..")

PGN_DIR = "Data/pgn/"
DATA_DIR = "Data/self_play.csv"
GAME_BATCH_SIZE = 1
CLAIM_DRAW = True
ENGINE_NAME = "MINIMAX"


def write_board_data(boards, moves, result):
    with open(DATA_DIR, "a") as f:
        player = 1
        for i in range(len(boards)):
            board = boards[i]
            move = moves[i]
            curr_result = []
            for row in board:
                for piece in row:
                    if piece == 1:
                        curr_result.append(1)
                    else:
                        curr_result.append(0)
            for row in board:
                for piece in row:
                    if piece == -1:
                        curr_result.append(1)
                    else:
                        curr_result.append(0)
            curr_result = curr_result + [ 0 if player == 1 else 1 for _ in range(5*5)]
            curr_result.append("(" + move + ")")
            curr_result.append(0)
            f.write(','.join(map(str, curr_result))+"\n")
            player = -player


def play_game():
    boards = []
    moves = []
    prev_board = None
    board, player, remain_time_o, remain_time_x = engine.initVariables()
    move_count = 0
    while not bool(engine.game_over(board)) and not move_count >= 500:
        begin = time.time()
        score, best_move = engine.move(prev_board, board, player, remain_time_x, remain_time_o)
        prev_board = deepcopy(board)
        boards.append(prev_board)
        moves.append(''.join([''.join(map(str, x)) for x in best_move])+":"+str(player*score))
        engine.update_board(board, best_move)
        move_count += 1
        player = -player
        time_elapsed = time.time() - begin
        print("Time elapsed from start a move to next: " + str(time_elapsed))
    result = ("1-0" if engine.game_over(board) == 1 else "0-1") if move_count < 500 else "1/2-1/2"
    print(result)
    print(board)
    engine.print_board(board)

    write_board_data(boards, moves, result)
    # write_game_data(game)


def main():
    for i in range(GAME_BATCH_SIZE):
        play_game()


if __name__ == "__main__":
    main()
