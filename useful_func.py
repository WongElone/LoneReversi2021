import numpy as np

def state_to_x(board):
    x = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:  # if occupied by black
                x += [1, 0]
            elif board[i][j] == 1:  # if occupied by white
                x += [0, 1]
            else:
                x += [0, 0]
    return np.array(x)

def create_actions_list():
    actions = []
    for i in range(8):
        for j in range(8):
            if (i, j) not in {(3, 3), (3, 4), (4, 3), (4, 4)}:
                actions.append((i, j))
    return actions

action_list = create_actions_list()

def find_action_idx(actions, action_list):
    action_idx = [action_list.index(action) for action in actions]
    return action_idx

def ai_move(y, legal_moves, action_list):
    # what is the format of y
    #   a (1, 60) array
    prioritized_action_idx = reversed(np.argsort(y[0]))
    possible_actions = legal_moves
    possible_actions_idx = find_action_idx(possible_actions, action_list)
    prioritized_possible_action_idx = []
    for item in prioritized_action_idx:
        if item in possible_actions_idx:
            prioritized_possible_action_idx.append(item)

    confidence = 0.9
    num_moves = len(prioritized_possible_action_idx)
    for _ in range(num_moves):
        if len(prioritized_possible_action_idx) == 1  or  np.random.rand() < confidence:
            return action_list[prioritized_possible_action_idx[0]]
        prioritized_possible_action_idx.pop(0)

def random_move(legal_moves):
    return legal_moves[np.random.choice(len(legal_moves))]



