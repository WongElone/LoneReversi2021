from board import Reversi
import numpy as np
from tensorflow.keras.models import load_model

Rev = Reversi()
Rev.reset()

def state_to_x():
    x = []
    for i in range(8):
        for j in range(8):
            if Rev.board[i][j] == 0:  # if occupied by black
                x += [1, 0]
            elif Rev.board[i][j] == 1:  # if occupied by white
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

def find_action_idx(actions):
    action_idx = []
    for action in actions:
        action_idx.append(action_list.index(action))
    return action_idx

def ai_move(y):
    # what is the format of y
    #   a (1, 60) array
    prioritized_action_idx = reversed(np.argsort(y[0]))
    possible_actions = Rev.legal_moves(Rev.turn)
    possible_actions_idx = find_action_idx(possible_actions)
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

def random_move():
    possible_actions = Rev.legal_moves(Rev.turn)
    return possible_actions[np.random.choice(len(possible_actions))]


### main ###
print("\n  test mode 1: model VS random")
print("  test mode 2: model VS model")
while 1:
    test_mode = input("\n  select test mode: ")
    if test_mode == '1' or test_mode == '2':
        break
    else:
        print("  type 1 or 2 and then press enter, try again")

if test_mode == '1':
    while 1:
        model_choice = input("\n  select a model to test (black/white): ")
        if model_choice == 'black' or model_choice == 'white':
            break
        else:
            print("  type 'black' or 'white' and then press enter, try again")

    num_games = '1'
    while 1:
        num_games = input("\n  input number of games to be played: ")
        if len(num_games) != 0:
            input_again = False
            for i in range(len(num_games)):
                if num_games[i] not in '0123456789':
                    input_again = True
                    break
            if not input_again:
                break
        print("  type a positive number and then press enter, try again")

    ai_side = None
    model = None
    if model_choice == 'black':
        ai_side = False
        model = load_model('M1.h5')
    else:
        ai_side = True
        model = load_model('M2.h5')
    win_count = [0, 0, 0]    # black count, white count, draw count
    for _ in range(int(num_games)):
        Rev.reset()
        while 1:
            if Rev.game_over():
                whowin = Rev.check_whowin()
                win_count[whowin] += 1
                break
            if Rev.turn == ai_side:
                x = state_to_x()
                y = model.predict(np.expand_dims(x, axis=0))
                Rev.update_board(ai_move(y), Rev.turn)
            else:
                Rev.update_board(random_move(), Rev.turn)

    print("\n  test mode:", test_mode)
    print("  AI's side:", model_choice)
    print("  AI's win-rate:", win_count[int(ai_side)]/sum(win_count))
    print("\n  number of games played: " + num_games)
    print("  num of black won:", win_count[0])
    print("  num of white won:", win_count[1])
    print("  num of draw:", win_count[2])

