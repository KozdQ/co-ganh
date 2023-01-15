import threading
from random import random
from math import inf
from copy import deepcopy


def blocked(board, x, y) -> bool:
    size = len(board)
    return x < 0 or x >= size or y < 0 or y >= size or board[x][y] != 0


def diff_cell(board, player, xC, yC) -> bool:
    size = len(board)
    return xC < 0 or xC >= size or yC < 0 or yC >= size or board[xC][yC] == -player


def legal_moves(prev_board, board, player):
    moves = []  # ((xS, yS), (xE, yE))
    trap_moves = []
    trapped = False
    sizeBoard = len(board)
    for row in range(sizeBoard):
        for col in range(sizeBoard):
            if prev_board is not None and board[row][col] == 0 and prev_board[row][col] == -player:
                xE = row
                yE = col
                pair_cells = []
                if (xE + yE) % 2 == 0:
                    pair_cells = [((xE - 1, yE), (xE + 1, yE)), ((xE, yE - 1), (xE, yE + 1)),
                                  ((xE - 1, yE + 1), (xE + 1, yE - 1)), ((xE - 1, yE - 1), (xE + 1, yE + 1))]
                else:
                    pair_cells = [((xE - 1, yE), (xE + 1, yE)), ((xE, yE - 1), (xE, yE + 1))]
                for first_cell, second_cell in pair_cells:
                    if isTrap(prev_board, board, -player, first_cell, second_cell):
                        cover_cells = []
                        if (xE + yE) % 2 == 0:
                            cover_cells = [(xE - 1, yE), (xE + 1, yE), (xE, yE - 1), (xE, yE + 1), (xE - 1, yE - 1),
                                           (xE - 1, yE + 1), (xE + 1, yE - 1), (xE + 1, yE + 1)]
                        else:
                            cover_cells = [(xE - 1, yE), (xE + 1, yE), (xE, yE - 1), (xE, yE + 1)]

                        for xS, yS in cover_cells:
                            if not diff_cell(board, player, xS, yS) and board[xS][yS] == player:
                                trapped = True
                                trap_moves.append(((xS, yS), (xE, yE)))

            if not trapped and board[row][col] == player:
                xS = row
                yS = col
                draft_moves = []
                if (xS + yS) % 2 == 0:
                    draft_moves = [(xS - 1, yS), (xS + 1, yS), (xS, yS - 1), (xS, yS + 1), (xS - 1, yS - 1),
                                   (xS - 1, yS + 1), (xS + 1, yS - 1), (xS + 1, yS + 1)]
                else:
                    draft_moves = [(xS - 1, yS), (xS + 1, yS), (xS, yS - 1), (xS, yS + 1)]
                for xE, yE in draft_moves:
                    if not blocked(board, xE, yE):
                        moves.append(((xS, yS), (xE, yE)))

    if trapped and len(trap_moves) > 0:
        return trap_moves
    return moves


def game_over(board):
    count_x = 0
    count_o = 0
    for row in board:
        for piece in row:
            if piece == -1:
                count_x += 1
            elif piece == 1:
                count_o += 1
    if count_x == 0:
        return -1
    elif count_o == 0:
        return 1
    else:
        return 0


def game_score(board, player, check_game):
    if check_game != 0:
        return player * sum([sum(x) for x in board])
    return random() + player * sum([sum(x) for x in board])


def isTrap(prev_board, board, player, first_cell, second_cell):
    if not diff_cell(board, 0, first_cell[0], first_cell[1]) and not diff_cell(board, 0, second_cell[0],
                                                                               second_cell[1]):
        if board[first_cell[0]][first_cell[1]] == player and board[first_cell[0]][first_cell[1]] == \
                board[second_cell[0]][second_cell[1]]:
            return True
    return False


def check_ganh_and_update(board, player, first_cell, second_cell):
    if not diff_cell(board, 0, first_cell[0], first_cell[1]) and not diff_cell(board, 0, second_cell[0],
                                                                               second_cell[1]):
        if board[first_cell[0]][first_cell[1]] == board[second_cell[0]][second_cell[1]]:
            board[first_cell[0]][first_cell[1]] = player
            board[second_cell[0]][second_cell[1]] = player


def check_vay_and_update(board, player, x, y, visited, need_update):
    visited[x][y] = True
    need_update.append((x, y))
    cover_cells = []
    if (x + y) % 2 == 0:
        cover_cells = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1),
                       (x + 1, y + 1)]
    else:
        cover_cells = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

    friend = []
    for xC, yC in cover_cells:
        if not diff_cell(board, player, xC, yC):
            if board[xC][yC] == 0:
                return False
            else:
                if (xC, yC) not in need_update:
                    friend.append((xC, yC))
    if len(friend) > 0:
        for xC, yC in friend:
            if not check_vay_and_update(board, player, xC, yC, visited, need_update):
                return False
    return True


def update_cells(board, player, cells):
    for x, y in cells:
        board[x][y] = player


def update_board(board, legal_move):
    if legal_move == None:
        return

    xS, yS = legal_move[0]
    xE, yE = legal_move[1]
    if board[xS][yS] == 0:
        # ("UPDATE START CELL ERROR", xS, yS, xE, yE)
        return
    if board[xE][yE] != 0:
        # ("UPDATE END CELL ERROR", xS, yS, xE, yE)
        return

    # Basic update
    player = board[xS][yS]
    board[xE][yE] = player
    board[xS][yS] = 0

    # Ganh
    pair_cells = []
    if (xE + yE) % 2 == 0:
        pair_cells = [((xE - 1, yE), (xE + 1, yE)), ((xE, yE - 1), (xE, yE + 1)), ((xE - 1, yE + 1), (xE + 1, yE - 1)),
                      ((xE - 1, yE - 1), (xE + 1, yE + 1))]
    else:
        pair_cells = [((xE - 1, yE), (xE + 1, yE)), ((xE, yE - 1), (xE, yE + 1))]
    for first_cell, second_cell in pair_cells:
        check_ganh_and_update(board, player, first_cell, second_cell)

    # Vay
    size = len(board)
    visited = [[False for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            if not visited[row][col] and board[row][col] == -player:
                need_update = []
                if check_vay_and_update(board, -player, row, col, visited, need_update):
                    update_cells(board, player, need_update)


def minimax(prev_board, board, player, vr_player, depth, event, alpha=-inf, beta=inf, ):
    check_game = game_over(board)
    if event.is_set() or depth == 0 or bool(check_game):
        return (game_score(board, player, check_game), [None])

    moves = legal_moves(prev_board, board, vr_player)

    # -------
    if len(moves) == 0:
        return (-player * vr_player * 16, [None])
        # ("BI VAY, next_move: ", vr_player)
    # -------

    if vr_player == player:
        max_score, best_move = -inf, [None]
        for legal_move in moves:
            vr_board = deepcopy(board)
            update_board(vr_board, legal_move)
            score, next_move = minimax(board, vr_board, player, -vr_player, depth - 1, event, alpha, beta)
            alpha = max(alpha, score)
            if score > max_score:
                max_score = score
                best_move = [legal_move] + next_move
            elif score == max_score:
                new_best_move = [legal_move] + next_move
                if len(new_best_move) < len(best_move):
                    best_move = new_best_move
                    max_score = score
            if beta <= alpha:
                break

        return (max_score, best_move)
    else:
        min_score, best_move = inf, [None]
        for legal_move in moves:
            vr_board = deepcopy(board)
            update_board(vr_board, legal_move)
            score, next_move = minimax(board, vr_board, player, -vr_player, depth - 1, event, alpha, beta)
            beta = min(beta, score)
            if score < min_score:
                min_score = score
                best_move = [legal_move] + next_move
            elif score == min_score:
                new_best_move = [legal_move] + next_move
                if len(new_best_move) < len(best_move):
                    best_move = new_best_move
                    min_score = score
            if beta <= alpha:
                break

        return (min_score, best_move)


def set_timeout(event):
    event.set()


def move(prev_board, board, player, remain_time_x, remain_time_o):
    event = threading.Event()
    s_time_value = 2.9
    t = threading.Timer(s_time_value, set_timeout, [event])
    t.start()

    depth = 4
    score, best_moves = minimax(prev_board, board, player, player, depth, event)

    t.cancel()

    if best_moves is None or best_moves[0] is None:
        return None
    return score, best_moves[0]


def initVariables():
    initBoard = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, -1],
        [-1, 0, 0, 0, -1],
        [-1, -1, -1, -1, -1]
    ]  # init board
    player = 1  # O play first
    remain_time_o = 600  # 10 minutes
    remain_time_x = 600  # 10 minutes

    return initBoard, player, remain_time_o, remain_time_x


def print_board(board):
    for row in list(reversed(board)):
        for piece in row:
            if piece == 1:
                piece = "X"
            elif piece == -1:
                piece = "O"
            else:
                piece = "-"
            print(str(piece).rjust(2, ' '), end=" ")
        print()
    print()
