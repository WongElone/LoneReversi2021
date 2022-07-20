import numpy as np
from copy import deepcopy
import time
from board import Reversi

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
            if (ii, jj) == (i,
                            j): continue  # this ignore the square just occupied by the latest move, since it should be removed from neighbors
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


def static_eval(board):
    pass

pos_cache = {}
# pos_cache's format: {move_th:{possible_pos_code:[reachable_next_pos_codes...],...},...}

# position's format: (board, turn, neighbors)
def minimax_eval(maxim_side, position, depth, pos_cache, alpha=float('-inf'), beta=float('inf')):
    board, turn, neighbors = position
    if depth == 0 or game_over(board, neighbors):
        maxim_side_points = len(np.where(board == int(maxim_side))[0])
        opponent_points = len(np.where(board == int(not maxim_side))[0])
        return float(maxim_side_points - opponent_points)

    children = []  # children are possible next positions
    code = pos_to_code(position)
    move_th = sum(board.flatten() != 2) - 4
    label_1, label_2 = False, False
    if move_th in pos_cache.keys():
        label_1 = True
        if code in pos_cache[move_th].keys():
            children = []    # not sure if this line is needed or not
            label_2 = True
            children_code = pos_cache[move_th][code]
            for temp in children_code:
                children.append(code_to_pos(temp))
    if not label_1:
        pos_cache[move_th] = {}
    if not label_2:
        children = []    # not sure if this line is needed or not
        children_code = []
        possible_moves = legal_moves(board, neighbors, turn)
        for move in possible_moves:
            temp = new_position(board, neighbors, move, turn)
            children.append(temp)
            children_code.append(pos_to_code(temp))
        pos_cache[move_th][code] = children_code

    if turn == maxim_side:
        max_eval = float('-inf')
        for child in children:
            eval = minimax_eval(maxim_side, child, depth - 1, pos_cache, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval

    else:
        min_eval = float('inf')
        for child in children:
            eval = minimax_eval(maxim_side, child, depth - 1, pos_cache, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# pos_cache's format: {move_th:{possible_pos_code:[reachable_next_pos_codes...],...},...}
def update_cache(cache, move_th, chosen_next_pos_code):    # remove unreachable branches
    offsprings_of_chosen_next_pos_code = cache[move_th + 1][chosen_next_pos_code]

    ### remove all next positions codes except the chosen one from cache
    print("")
    print(cache)
    print("")
    # print(len(cache[1]))
    # print(len(cache[2]))
    # print(len(cache[3]))
    # print(len(cache[4]))
    # print(len(cache[5]))
    # print(move_th)
    possi_next_pos_codes = list(cache[move_th + 1].keys())
    for temp in possi_next_pos_codes:
        if temp != chosen_next_pos_code:
            del cache[move_th + 1][temp]
    del possi_next_pos_codes

    print(cache)
    ### remove all unreachable offsprings pos from cache###
    move_th_tracker = move_th + 2
    loop_label = move_th_tracker in cache.keys()
    while loop_label:
        possi_pos_codes = list(cache[move_th_tracker].keys())    # no need deepcopy because list(dict_keys) return a new reference
        for temp in possi_pos_codes:
            if temp not in offsprings_of_chosen_next_pos_code:
                del cache[move_th_tracker][temp]
        del possi_pos_codes

        if move_th_tracker + 1 not in cache.keys():
            loop_label = False
            continue

        temp_offsprings_code = deepcopy(offsprings_of_chosen_next_pos_code)
        offsprings_of_chosen_next_pos_code = []
        # print("")
        # print(cache)
        # print(move_th_tracker)
        # print(cache[move_th_tracker])
        # print(temp_offsprings_code)
        # print("")
        for temp in temp_offsprings_code:
            # print(cache[move_th_tracker][temp])
            offsprings_of_chosen_next_pos_code += cache[move_th_tracker][temp]
        del temp_offsprings_code
        move_th_tracker += 1

def minimax_move(maxim_side, depth, possible_moves, board, neighbors, turn, cache):
    eval_list = []
    for move in possible_moves:
        next_position = new_position(board, neighbors, move, turn)
        eval_list.append(minimax_eval(maxim_side, next_position, depth, cache))
    return possible_moves[np.argmax(eval_list)]

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

    return (board, turn, neighbors)



if __name__ == "__main__":
    Rev = Reversi()
    Rev.reset()

    # rand_board = np.random.randint(low=0, high=3, size=(8, 8))
    # rand_turn = bool(np.random.randint(low=0, high=2))
    # # rand_neighbors_num = np.random.randint(low=0, high=29)
    # rand_neighbors_num = 10
    #
    # rand_neighbors = []
    # for _ in range(rand_neighbors_num):
    #     while 1:
    #         i = np.random.randint(low=0, high=8)
    #         j = np.random.randint(low=0, high=8)
    #         if (i, j) not in rand_neighbors:
    #             rand_neighbors.append((i, j))
    #             break
    start_time = time.time()
    tree = {}
    position = Rev.board, Rev.turn, Rev.neighbors
    code = pos_to_code(position)
    for i in range(1000000):
        if i == 5123:
            tree[code] = {1, 2, 3, 4, 5}
            continue
        fake_code = list(deepcopy(code))
        fake_code[0] = np.uint32(np.random.randint(low=0, high= 1000000000))
        tree[tuple(fake_code)] = {1, 2, 3, 4 ,5, 6}
    print("data gen time:", time.time() - start_time)

    start_time = time.time()
    child = None
    for _ in range(10000):
        pos_code = pos_to_code(position)
        child = tree[pos_code]
    print("code search time:", time.time() - start_time)
    print(child)


    Rev.reset()
    start_time = time.time()
    for _ in range(10000):
        possible_moves = legal_moves(Rev.board, Rev.neighbors, Rev.turn)
        children = []
        for move in possible_moves:
            children.append(new_position(Rev.board, Rev.neighbors, move, Rev.turn))
    print("play search time:", time.time() - start_time)

    # board, turn, neighbors = code_to_pos(code)
    # print("")
    # print(rand_board)
    # print(type(rand_board))
    # print(board)
    # print(type(board))
    # print("")
    # print(rand_turn)
    # print(type(rand_turn))
    # print(turn)
    # print(type(turn))
    # print("")
    # print(rand_neighbors)
    # print(type(rand_neighbors))
    # print(neighbors)
    # print(type(neighbors))