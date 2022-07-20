import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

def check_legal(move, board, neighbors, myside):
    # this methods check if a move is legal for any side
    # move is a tuple (i, j) where i is the row index, j is the col index
    # myside is a boolean indicating which side you wanna check
    i, j = move
    if (i, j) not in neighbors:
        return False
    if board[i, j] != 2:
        return False
    # check Up
    if i > 1:  # requirement of Up legal
        Up = np.flip(
            board[:i, j].flatten())  # need to flip because we want the first element to be the closest one
        if len(np.where(Up == int(myside))[0]) != 0:  # check if there exist same color
            distance = np.where(Up == int(myside))[0][0]  # close_same is the distance of closest same color
            if distance > 0:  # check if it is not adjacent
                if set(Up[0:distance]) == {
                    int(not myside)}: return True  # check if only opposite color exist before closest same color
    # check Down
    if i < 6:
        Down = board[i + 1:, j].flatten()
        if len(np.where(Down == int(myside))[0]) != 0:
            distance = np.where(Down == int(myside))[0][0]
            if distance > 0:
                if set(Down[0:distance]) == {int(not myside)}: return True
    # check Left
    if j > 1:
        Left = np.flip(board[i, : j])
        if len(np.where(Left == int(myside))[0]) != 0:
            distance = np.where(Left == int(myside))[0][0]
            if distance > 0:
                if set(Left[0:distance]) == {int(not myside)}: return True
    # check Right
    if j < 6:
        Right = board[i, j + 1:]
        if len(np.where(Right == int(myside))[0]) != 0:
            distance = np.where(Right == int(myside))[0][0]
            if distance > 0:
                if set(Right[0:distance]) == {int(not myside)}: return True
    # check UpLeft
    if i > 1 and j > 1:
        size = min(i, j)
        square = board[(i - size): i, (j - size): j]
        UpLeft = np.flip(square.diagonal())
        if len(np.where(UpLeft == int(myside))[0]) != 0:
            distance = np.where(UpLeft == int(myside))[0][0]
            if distance > 0:
                if set(UpLeft[0:distance]) == {int(not myside)}: return True
    # check UpRight
    if i > 1 and j < 6:
        size = min(i, 7 - j)
        square = board[(i - size): i, (j + 1): (j + 1 + size)]
        UpRight = np.flipud(square).diagonal()
        if len(np.where(UpRight == int(myside))[0]) != 0:
            distance = np.where(UpRight == int(myside))[0][0]
            if distance > 0:
                if set(UpRight[0:distance]) == {int(not myside)}: return True
    # check DownLeft
    if i < 6 and j > 1:
        size = min(7 - i, j)
        square = board[(i + 1): (i + 1 + size), (j - size):j]
        DownLeft = np.flip(np.flipud(square).diagonal())
        if len(np.where(DownLeft == int(myside))[0]) != 0:
            distance = np.where(DownLeft == int(myside))[0][0]
            if distance > 0:
                if set(DownLeft[0:distance]) == {int(not myside)}: return True
    # check DownRight
    if i < 6 and j < 6:
        size = min(7 - i, 7 - j)
        square = board[(i + 1): (i + 1 + size), (j + 1): (j + 1 + size)]
        DownRight = square.diagonal()
        if len(np.where(DownRight == int(myside))[0]) != 0:
            distance = np.where(DownRight == int(myside))[0][0]
            if distance > 0:
                if set(DownRight[0:distance]) == {int(not myside)}: return True
    # not satisfying all above legal conditions
    return False

def legal_moves(board, neighbors, myside):
    LM = []
    for move in neighbors:
        if check_legal(move, board, neighbors, myside): LM.append(move)
    return LM

def new_turn(board, neighbors, current_turn):  # tells which side plays next turn, given current turn
    LM = legal_moves(board, neighbors, not current_turn)
    if len(LM) == 0: return current_turn
    return not current_turn

def new_neighbors(board, neighbors, move):
    i, j = move
    new_neighbors = []
    for ii in range(i - 1, i + 2):
        if ii < 0 or ii > 7: continue  # this prevent indexing outside the board
        for jj in range(j - 1, j + 2):
            if jj < 0 or jj > 7: continue  # this prevent indexing outside the board
            if (ii, jj) == (i, j): continue  # this ignore the square just occupied by the latest move, since it should be removed from neighbors
            if board[ii, jj] != 2: continue  # filter out the squares already occupied
            new_neighbors.append((ii, jj))
    temp_neighbors = deepcopy(neighbors)
    if len(new_neighbors) > 0:
        temp_neighbors.update(new_neighbors)
    temp_neighbors.remove((i, j))  # this remove the square just occupied by the latest move
    return temp_neighbors

def new_position(board, neighbors, move, myside):
    temp_board = deepcopy(board)
    # this method updates the board, neighbors, points, and turn
    i, j = move
    # update center
    temp_board[i, j] = int(myside)
    # update Up
    if i > 1:  # requirement of Up legal
        Up = np.flip(temp_board[:i, j].flatten())  # need to flip because we want the first element to be the closest one
        if len(np.where(Up == int(myside))[0]) != 0:  # check if there exist same color
            distance = np.where(Up == int(myside))[0][0]  # close_same is the distance of closest same color
            if distance > 0:  # check if it is not adjacent
                if set(Up[0:distance]) == {int(not myside)}:  # check if only opposite color exist before closest same color
                    temp_board[(i - distance): i, j] = int(myside)  # update color, tips: distance somehow indicates the number of pieces to change color
    # update Down
    if i < 6:
        Down = temp_board[i + 1:, j].flatten()
        if len(np.where(Down == int(myside))[0]) != 0:
            distance = np.where(Down == int(myside))[0][0]
            if distance > 0:
                if set(Down[0:distance]) == {int(not myside)}:
                    temp_board[(i + 1): (i + 1 + distance), j] = int(myside)
    # update Left
    if j > 1:
        Left = np.flip(temp_board[i, : j])
        if len(np.where(Left == int(myside))[0]) != 0:
            distance = np.where(Left == int(myside))[0][0]
            if distance > 0:
                if set(Left[0:distance]) == {int(not myside)}:
                    temp_board[i, (j - distance): j] = int(myside)
    # update Right
    if j < 6:
        Right = temp_board[i, j + 1:]
        if len(np.where(Right == int(myside))[0]) != 0:
            distance = np.where(Right == int(myside))[0][0]
            if distance > 0:
                if set(Right[0:distance]) == {int(not myside)}:
                    temp_board[i, (j + 1): (j + 1 + distance)] = int(myside)
    # update UpLeft
    if i > 1 and j > 1:
        size = min(i, j)
        square = temp_board[(i - size): i, (j - size): j]
        UpLeft = np.flip(square.diagonal())
        if len(np.where(UpLeft == int(myside))[0]) != 0:
            distance = np.where(UpLeft == int(myside))[0][0]
            if distance > 0:
                if set(UpLeft[0:distance]) == {int(not myside)}:
                    for k in range(distance):
                        temp_board[i - 1 - k, j - 1 - k] = int(myside)
    # update UpRight
    if i > 1 and j < 6:
        size = min(i, 7 - j)
        square = temp_board[(i - size): i, (j + 1): (j + 1 + size)]
        UpRight = np.flipud(square).diagonal()
        if len(np.where(UpRight == int(myside))[0]) != 0:
            distance = np.where(UpRight == int(myside))[0][0]
            if distance > 0:
                if set(UpRight[0:distance]) == {int(not myside)}:
                    for k in range(distance):
                        temp_board[i - 1 - k, j + 1 + k] = int(myside)
    # update DownLeft
    if i < 6 and j > 1:
        size = min(7 - i, j)
        square = temp_board[(i + 1): (i + 1 + size), (j - size):j]
        DownLeft = np.flip(np.flipud(square).diagonal())
        if len(np.where(DownLeft == int(myside))[0]) != 0:
            distance = np.where(DownLeft == int(myside))[0][0]
            if distance > 0:
                if set(DownLeft[0:distance]) == {int(not myside)}:
                    for k in range(distance):
                        temp_board[i + 1 + k, j - 1 - k] = int(myside)
    # update DownRight
    if i < 6 and j < 6:
        size = min(7 - i, 7 - j)
        square = temp_board[(i + 1): (i + 1 + size), (j + 1): (j + 1 + size)]
        DownRight = square.diagonal()
        if len(np.where(DownRight == int(myside))[0]) != 0:
            distance = np.where(DownRight == int(myside))[0][0]
            if distance > 0:
                if set(DownRight[0:distance]) == {int(not myside)}:
                    for k in range(distance):
                        temp_board[i + 1 + k, j + 1 + k] = int(myside)

    # get new neighbors
    next_neighbors = new_neighbors(temp_board, neighbors, move)
    # get new points
    # next_n_black = len(np.where(board == 0)[0])
    # next_n_white = len(np.where(board == 1)[0])
    # get next turn
    next_turn = new_turn(board, neighbors, myside)

    return (temp_board, next_turn, next_neighbors)

def game_over(board, neighbors):
    for move in neighbors:
        if check_legal(move, board, neighbors, False): return False  # False is the black side
        if check_legal(move, board, neighbors, True): return False  # True is the white side
    return True

# position's format: (board, turn, neighbors)
def minimax_eval(maxim_side, position, depth, branch, alpha=float('-inf'), beta=float('inf')):
    board, turn, neighbors = position
    g_over = game_over(board, neighbors)
    if depth == 0 or g_over:
        ### below 3 lines are from the simple eval version ###
        # maxim_side_points = float(sum(board.flatten() == int(maxim_side)))
        # opponent_points = float(sum(board.flatten() == int(not maxim_side)))
        # return float(maxim_side_points - opponent_points)

        ### complex version ###
        if g_over:
            PARAM_0 = 100000.0
            maxim_side_points = float(sum(board.flatten() == int(maxim_side)))
            opponent_points = float(sum(board.flatten() == int(not maxim_side)))
            return PARAM_0 * float(maxim_side_points - opponent_points)
        else:
            # A_pos = {(0, 2), (0, 5), (2, 0), (2, 7), (5, 0), (5, 7), (7, 2), (7, 5)}
            # B_pos = {(0, 3), (0, 4), (3, 0), (3, 7), (4, 0), (4, 7), (7, 3), (7, 4)}
            # C_pos = {(0, 1), (0, 6), (1, 0), (1, 7), (6, 0), (6, 7), (7, 1), (7, 6)}
            X_pos = {(1, 1), (1, 6), (6, 1), (6, 6)}
            Corner_pos = {(0, 0), (0, 7), (7, 0), (7, 7)}

            MS = int(maxim_side)
            MS_pos = np.where(board == MS)
            MS_pos = list(zip(MS_pos[0], MS_pos[1]))
            MS_n = len(MS_pos)
            MS_X_pos = [pos for pos in MS_pos if pos in X_pos]
            MS_Corner_pos = [pos for pos in MS_pos if pos in Corner_pos]

            OP = int(not maxim_side)
            OP_pos = np.where(board == OP)
            OP_pos = list(zip(OP_pos[0], OP_pos[1]))
            OP_n = len(OP_pos)
            OP_X_pos = [pos for pos in OP_pos if pos in X_pos]
            OP_Corner_pos = [pos for pos in OP_pos if pos in Corner_pos]

            score = 0.0

            # number of possible move of both sides
            PARAM_1 = 0.5

            MS_possible_moves = legal_moves(board, neighbors, maxim_side)
            OP_possible_moves = legal_moves(board, neighbors, not maxim_side)
            score += PARAM_1 * (len(MS_possible_moves) - len(OP_possible_moves))

            # X position
            PARAM_5 = 15.0

            for pos in MS_X_pos:
                if pos == (1, 1):
                    if board[0, 0] == 2:
                        score -= PARAM_5
                    elif board[0, 0] == OP:
                        if set(np.diagonal(board)[1:7]) == {MS} and board[7, 7] == 2:
                            score -= PARAM_5
                elif pos == (1, 6):
                    if board[0, 7] == 2:
                        score -= PARAM_5
                    elif board[0, 7] == OP:
                        if set(np.diagonal(np.flipud(board))[1:7]) == {MS} and board[7, 0] == 2:
                            score -= PARAM_5
                elif pos == (6, 1):
                    if board[7, 0] == 2:
                        score -= PARAM_5
                    elif board[7, 0] == OP:
                        if set(np.diagonal(np.flipud(board))[1:7]) == {MS} and board[0, 7] == 2:
                            score -= PARAM_5
                else:
                    if board[7, 7] == 2:
                        score -= PARAM_5
                    elif board[7, 7] == OP:
                        if set(np.diagonal(board)[1:7]) == {MS} and board[0, 0] == 2:
                            score -= PARAM_5

            for pos in OP_X_pos:
                if pos == (1, 1):
                    if board[0, 0] == 2:
                        score -= -PARAM_5
                    elif board[0, 0] == MS:
                        if set(np.diagonal(board)[1:7]) == {OP} and board[7, 7] == 2:
                            score -= -PARAM_5
                elif pos == (1, 6):
                    if board[0, 7] == 2:
                        score -= -PARAM_5
                    elif board[0, 7] == MS:
                        if set(np.diagonal(np.flipud(board))[1:7]) == {OP} and board[7, 0] == 2:
                            score -= -PARAM_5
                elif pos == (6, 1):
                    if board[7, 0] == 2:
                        score -= -PARAM_5
                    elif board[7, 0] == MS:
                        if set(np.diagonal(np.flipud(board))[1:7]) == {OP} and board[0, 7] == 2:
                            score -= -PARAM_5
                else:
                    if board[7, 7] == 2:
                        score -= -PARAM_5
                    elif board[7, 7] == MS:
                        if set(np.diagonal(board)[1:7]) == {OP} and board[0, 0] == 2:
                            score -= -PARAM_5

            # corner
            PARAM_6a = 20.0

            score += PARAM_6a * (len(MS_Corner_pos) - len(OP_Corner_pos))

            # corner edge (edge sticked to corner), hanging edge, ABC position
            PARAM_6b = 8.0
            PARAM_6c = 4.0

            for edge in (board[0,:], board[7,:], board[:,0], board[:,7]):
                if set(edge) == {2}: continue
                edge_copy = deepcopy(edge)
                goto_6, goto_5, goto_4, goto_3, goto_2, goto_1 = True, True, True, True, True, True
                ### corner edge (edge sticked to corner) ###
                if not (edge_copy[0] == edge_copy[7] == 2):
                    if edge_copy[0] != 2:
                        sign = (-1) ** (int(edge_copy[0] == MS) + 1)
                        k = 0
                        while edge_copy[k] == edge_copy[k+1] and k < 6:
                            edge_copy[k] = 2
                            k += 1
                        edge_copy[k] = 2
                        score += sign * PARAM_6b * k
                    if edge_copy[7] != 2:
                        sign = (-1) ** (int(edge_copy[7] == MS) + 1)
                        k = 0
                        while edge_copy[7-k] == edge_copy[7-k-1] and k < 6:
                            edge_copy[7-k] = 2
                            k += 1
                        edge_copy[7-k] = 2
                        score += sign * PARAM_6b * k

                    if sum(edge_copy == MS) == 6 or sum(edge_copy == OP) == 6:
                        goto_5, goto_4, goto_3, goto_2, goto_1 = False, False, False, False, False
                    elif sum(edge_copy == MS) == 5 or sum(edge_copy == OP) == 5:
                        goto_6 = False
                    elif sum(edge_copy == MS) == 4 or sum(edge_copy == OP) == 4:
                        goto_6, goto_5 = False, False
                    elif sum(edge_copy == MS) == 3 or sum(edge_copy == OP) == 3:
                        goto_6, goto_5, goto_4 = False, False, False
                    elif sum(edge_copy == MS) == 2 or sum(edge_copy == OP) == 2:
                        goto_6, goto_5, goto_4, goto_3 = False, False, False, False
                    elif sum(edge_copy == MS) == 1 or sum(edge_copy == OP) == 1:
                        goto_6, goto_5, goto_4, goto_3, goto_2 = False, False, False, False, False
                    else:
                        continue
                ### hanging edge ###
                if goto_6:
                    temp = set(edge_copy[1:7])
                    if temp == {MS}:
                        if edge[0] == edge[7] == OP:
                            score += PARAM_6b * 6
                        else:
                            score += PARAM_6c * 6
                        edge_copy[1:7] = 2
                        continue
                    elif temp == {OP}:
                        if edge[0] == edge[7] == MS:
                            score += -PARAM_6b * 6
                        else:
                            score += -PARAM_6c * 6
                        edge_copy[1:7] = 2
                        continue
                if goto_5:
                    found_all = False
                    for k in range(1, 3):
                        if found_all: continue
                        temp = set(edge_copy[k:k+5])
                        if temp == {MS}:
                            if edge[k-1] == edge[k+5] == OP:
                                score += PARAM_6b * 5
                            else:
                                score += PARAM_6c * 6
                            edge_copy[k:k+5] = 2
                            found_all = True
                            goto_4, goto_3, goto_2 = False, False, False
                        elif temp == {OP}:
                            if edge[k-1] == edge[k+5] == MS:
                                score += -PARAM_6b * 5
                            else:
                                score += -PARAM_6c * 6
                            edge_copy[k:k + 5] = 2
                            found_all = True
                            goto_4, goto_3, goto_2 = False, False, False
                if goto_4:
                    found_all = False
                    for k in range(1, 4):
                        if found_all: continue
                        temp = set(edge_copy[k:k+4])
                        if temp == {MS}:
                            if edge[k-1] == edge[k+4] == OP:
                                score += PARAM_6b * 4
                            else:
                                score += PARAM_6c * 4
                            edge_copy[k:k + 4] = 2
                            found_all = True
                            goto_3 = False
                        elif temp == {OP}:
                            if edge[k-1] == edge[k+4] == MS:
                                score += -PARAM_6b * 4
                            else:
                                score += -PARAM_6c * 4
                            edge_copy[k:k + 4] = 2
                            found_all = True
                            goto_3 = False
                if goto_3:
                    found_all = False
                    for k in range(1, 5):
                        if found_all: continue
                        temp = set(edge_copy[k:k+3])
                        if temp == {MS}:
                            if edge[k-1] == edge[k+3] == OP:
                                score += PARAM_6b * 3
                            else:
                                score += PARAM_6c * 3
                            edge_copy[k:k + 3] = 2
                            if k!= 1:
                                found_all = True
                        elif temp == {OP}:
                            if edge[k-1] == edge[k+3] == MS:
                                score += -PARAM_6b * 3
                            else:
                                score += -PARAM_6c * 3
                            edge_copy[k:k + 3] = 2
                            if k!= 1:
                                found_all = True
                if goto_2:
                    for k in range(1, 5):
                        temp = set(edge_copy[k:k+2])
                        if temp == {MS}:
                            if edge[k-1] == edge[k+2] == OP:
                                score += PARAM_6b * 2
                            else:
                                score += PARAM_6c * 2
                            edge_copy[k:k+2] = 2
                        elif temp == {OP}:
                            if edge[k-1] == edge[k+2] == MS:
                                score += -PARAM_6b * 2
                            else:
                                score += -PARAM_6c * 2
                            edge_copy[k:k+2] = 2
                if goto_1:
                    for k in range(1, 6):
                        temp = edge_copy[k]
                        if temp == MS:
                            if edge[k-1] == edge[k+1] == OP:
                                score += PARAM_6b
                            else:
                                score += PARAM_6c
                        elif temp == OP:
                            if edge[k-1] == edge[k+1] == MS:
                                score += -PARAM_6b
                            else:
                                score += -PARAM_6c

            # points difference
            PARAM_7 = 618 * float(np.exp(0.618 * (MS_n + OP_n - 64)))

            score += PARAM_7 * (MS_n - OP_n)

            return score
    ### eval end ###

    children = []  # children are possible next positions
    pos_code = pos_to_code(position)
    label_1, label_2 = False, False
    if pos_code in branch.keys():
        label_1 = True
        if bool(branch[pos_code]):    # if branch[pos_code] is not an empty {}
            label_2 = True
            children_code = list(branch[pos_code].keys())
            for temp in children_code:
                children.append(code_to_pos(temp))
    if not label_1:
        branch[pos_code] = {}
    if not label_2:
        possible_moves = legal_moves(board, neighbors, turn)
        for move in possible_moves:
            child_pos = new_position(board, neighbors, move, turn)
            children.append(child_pos)
            branch[pos_code][pos_to_code(child_pos)] = {}

    if turn == maxim_side:
        max_eval = float('-inf')
        for child in children:
            eval = minimax_eval(maxim_side, child, depth - 1, branch[pos_code], alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval

    else:
        min_eval = float('inf')
        for child in children:
            eval = minimax_eval(maxim_side, child, depth - 1, branch[pos_code], alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def minimax_move(maxim_side, depth, possible_moves, board, neighbors, turn, cache, multiproc):
    group_next_position = [new_position(board, neighbors, move, turn) for move in possible_moves]
    if multiproc:
        group_maxim_side = [maxim_side for _ in range(len(group_next_position))]
        group_depth = [depth for _ in range(len(group_next_position))]
        group_cache = [cache for _ in range(len(group_next_position))]
        with ProcessPoolExecutor() as executor:
            eval_list = list(executor.map(minimax_eval, group_maxim_side, group_next_position, group_depth, group_cache))
    else:
        eval_list = [minimax_eval(maxim_side, next_position, depth, cache) for next_position in group_next_position]
    return possible_moves[np.argmax(eval_list)], max(eval_list)

# position's format: (board, turn, neighbors)
def pos_to_code(position):
    board, turn, neighbors = position

    board_ternary = board.flatten()
    b_id = []
    for i in range(4):
        section_id = 0
        for count, value in enumerate(board_ternary[i * 16: (i + 1) * 16], start=1):
            section_id += value * (3 ** (16 - count))
        b_id.append(section_id)

    neighbors_binary = np.zeros(64, dtype=np.uint32)
    for pos in neighbors:
        neighbors_binary[pos[0] * 8 + pos[1]] = np.uint32(1)
    n_id = []
    for i in range(2):
        section_id = np.uint32(0)
        for count, value in enumerate(neighbors_binary[i * 32: (i + 1) * 32], start=1):
            section_id += np.uint32(value * (2 ** (32 - count)))
        n_id.append(section_id)

    code = b_id + [turn] + n_id
    return tuple(code)

# code's format: [b_id 1, b_id 2, b_id 3, b_id 4, turn, n_id 1, n_id 2]
def code_to_pos(code):
    b_id, turn, n_id = code[:4], code[4], code[5:]

    board_ternary = []
    for section in range(4):
        section_ternary = []
        n = b_id[section]
        while n:
            n, r = divmod(n, 3)
            section_ternary.insert(0, r)
        while len(section_ternary) < 16:
            section_ternary.insert(0, 0)
        board_ternary += section_ternary
    board = np.array(board_ternary, dtype=int).reshape(8, 8)

    neighbors_binary = []
    for section in range(2):
        section_binary = []
        n = n_id[section]
        while n:
            n, r = divmod(n, 2)
            section_binary.insert(0, int(r))
        while len(section_binary) < 32:
            section_binary.insert(0, 0)
        neighbors_binary += section_binary
    neighbors = []
    for count, value in enumerate(neighbors_binary):
        if value:
            neighbors.append((count // 8, count % 8))
    neighbors = set(neighbors)

    return (board, turn, neighbors)
