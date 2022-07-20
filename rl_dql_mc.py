from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
import numpy as np
import board
import time
import useful_func

Rev = board.Reversi()


class DQN:
    def __init__(self, D, K, H, inherit = False, old_model = None):
        self.D = D    # D: number of input nodes
        self.K = K    # K: num of output nodes
        self.H = H    # H should be a list of hidden layer sizes

        self.model = old_model
        if not inherit:
            I = Input(shape=(D,))
            A = I
            for i in range(len(H)):
                A = Dense(units=H[i], activation='relu')(A)
            O = Dense(units=K)(A)
            model = Model(I, O)
            model.compile(optimizer='Adam', loss='mse')
            self.model = model

    def shuffle_in_unison(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def gen_train_data(self, replay_buffer):
        # current episode or play_data or replay_buffer 's format: [[x, possible actions idx, performed actions idx, return], ...]

        X_train = []
        Y_train = []

        # always learn the episode just finished
        play_data = []

        # also randomly choose a fixed number of episodes from the replay buffer
        for i in range(len(replay_buffer)):
        ###   randomly choose some pairs from the i th episode   ###
            num_sample_per_episode = 16
            if len(replay_buffer[i]) < num_sample_per_episode:
                num_sample_per_episode = len(replay_buffer[i])
            idx = np.random.choice(len(replay_buffer[i]), size=num_sample_per_episode, replace=False)
            play_data += [replay_buffer[i][j] for j in idx]

        for i in range(len(play_data)):
            Y_hat = self.model.predict(np.expand_dims(play_data[i][0], axis=0))[0]
            Y = np.zeros((self.K,)).astype(float)

            for j in range(self.K):
                if j not in play_data[i][1]:    # check if action j not in possible moves in the i th turn of the previous episode
                    Y[j] = -100    # assign -100 to Y[j] to punish impossible moves
                elif j != play_data[i][2]:
                    Y[j] = Y_hat[j]    # assign predicted return to Y[j] for unselected moves
                elif j == play_data[i][2]:
                    Y[j] = play_data[i][3]    # assign return to Y[j] for the selected move
                else:
                    print('error: type 0')    # it is an error for reaching here
            # current episode or play_data or replay_buffer 's format: [[x, possible actions idx, performed actions idx, return], ...]
            X_train.append(play_data[i][0])
            Y_train.append(Y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        self.shuffle_in_unison(X_train, Y_train)

        return X_train, Y_train

    def train(self, replay_buffer, show_loss):
        X_train, Y_train = self.gen_train_data(replay_buffer)
        self.model.fit(X_train, Y_train, epochs=10, verbose=int(show_loss))


action_list = useful_func.create_actions_list()

def choose_action(prioritized_actions_idx, possible_actions_idx, iter, PRIOR_ITERS):
    best_action_idx = 0
    for action_idx in prioritized_actions_idx:
        if action_idx in possible_actions_idx:
            best_action_idx = action_idx
            break
    eps = 0.25
    if iter + PRIOR_ITERS > 1000:
        eps = 0.2
    elif iter + PRIOR_ITERS > 5000:
        eps = 0.15
    elif iter + PRIOR_ITERS > 25000:
        eps = 0.1
    elif iter + PRIOR_ITERS > 100000:
        eps = 0.05
    if np.random.rand() < eps:
        return possible_actions_idx[np.random.choice(len(possible_actions_idx))]
    return best_action_idx

def play_one_episode(iter):
    Rev.reset()
    black_record = []
    white_record = []
    while 1:
        x = useful_func.state_to_x(Rev.board)
        if Rev.turn:  # if white's turn
            if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
                y = dqn_white.model.predict(np.expand_dims(x, axis=0))
            else:
                y = None
        else:
            if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
                y = dqn_black.model.predict(np.expand_dims(x, axis=0))
            else:
                y = None

        possible_actions = Rev.legal_moves(Rev.turn)
        if y is not None:
            prioritized_actions_idx = reversed(np.argsort(y)[0])
            possible_actions_idx = useful_func.find_action_idx(possible_actions, action_list)

            chosen_action_idx = choose_action(prioritized_actions_idx, possible_actions_idx, iter, PRIOR_ITERS)
            chosen_action = action_list[chosen_action_idx]

            if Rev.turn:
                white_record.append([x, possible_actions_idx, chosen_action_idx])
            else:
                black_record.append([x, possible_actions_idx, chosen_action_idx])
        else:
            chosen_action = useful_func.random_move(possible_actions)

        Rev.update_board(chosen_action, Rev.turn)

        if Rev.game_over():
            # black_rewards = (1, -1, 0)
            # white_rewards = (-1, 1, 0)
            whowin = Rev.check_whowin()
            # black_reward = black_rewards[whowin]
            # white_reward = white_rewards[whowin]

            if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
                black_reward = Rev.n_black - Rev.n_white
                for t in range(len(black_record) - 1, -1, -1):
                    black_record[t].append(black_reward)
                    black_reward = GAMMA * black_reward
            if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
                white_reward = Rev.n_white - Rev.n_black
                for t in range(len(white_record) - 1, -1, -1):
                    white_record[t].append(white_reward)
                    white_reward = GAMMA * white_reward
            # now the format of black_record or white_record: [[x, possible actions idx, performed actions idx, return], ...]
            return black_record, white_record, whowin

### main ###
GAMMA = 0.95
NUM_ITERS = 300
EPISODES_PER_ITER = 10
REPLAY_BUFFER_SIZE = 50
SHOW_INTERVAL = 10
PRIOR_ITERS = 400

D = 128
K = 60
H = [1024, 1024, 1024]

print("\n  train mode 1: model VS random")
print("  train mode 2: model VS model")
while 1:
    train_mode = input("\n  select train mode: ")
    if train_mode == '1' or train_mode == '2':
        break
    else:
        print("  type 1 or 2 and then press enter, try again")

if train_mode == '1':
    while 1:
        model_choice = input("\n  train black model or white model? (B/W): ")
        if model_choice == 'B' or model_choice == 'W':
            break
        else:
            print("  type 'B' or 'W' and then press enter, try again")

while 1:
    inherit = input("\n  inherit old model? (y/n): ")
    if inherit == 'y':
        inherit = True
        break
    elif inherit == 'n':
        inherit = False
        break
    else:
        print("  type 'y' or 'n and then press enter, try again")

if inherit:
    if train_mode == '1':
        if model_choice == 'B':
            black_model = load_model('M1.h5')
        elif model_choice == 'W':
            white_model = load_model('M2.h5')
    else:
        black_model = load_model('M1.h5')
        white_model = load_model('M2.h5')
else:
    black_model = None
    white_model = None

if train_mode == '1':
    if model_choice == 'B':
        dqn_black = DQN(D, K, H, inherit, black_model)
    elif model_choice == 'W':
        dqn_white = DQN(D, K, H, inherit, white_model)
else:
    dqn_black = DQN(D, K, H, inherit, black_model)
    dqn_white = DQN(D, K, H, inherit, white_model)


black_replay_buffer = []
white_replay_buffer = []

black_win_count = 0
white_win_count = 0
draw_count = 0
whowin_to_count = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

start_time = time.time()
checkpoint_time = time.time()
for iter in range(NUM_ITERS):
    for _ in range(EPISODES_PER_ITER):
        black_record, white_record, whowin = play_one_episode(iter)
        black_win_count += whowin_to_count[whowin][0]
        white_win_count += whowin_to_count[whowin][1]
        draw_count += whowin_to_count[whowin][2]
        # train model if replay buffer len > certain number
        if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
            if len(black_replay_buffer) >= REPLAY_BUFFER_SIZE:
                # remove the oldest episode in replay buffer
                black_replay_buffer.pop(0)
            # add the current_episode
            black_replay_buffer.append(black_record)
        if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
            if len(white_replay_buffer) >= REPLAY_BUFFER_SIZE:
                # remove the oldest episode in replay buffer
                white_replay_buffer.pop(0)
            # add the current_episode
            white_replay_buffer.append(white_record)
    show_loss = (iter % SHOW_INTERVAL == 0)
    if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
        if len(black_replay_buffer) == REPLAY_BUFFER_SIZE:
            dqn_black.train(black_replay_buffer, show_loss)
    if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
        if len(white_replay_buffer) == REPLAY_BUFFER_SIZE:
            dqn_white.train(white_replay_buffer, show_loss)

### presenting the win count and iterations passed ###
    if iter % SHOW_INTERVAL == SHOW_INTERVAL - 1:
        print("\nepisodes played:", (iter + 1) * EPISODES_PER_ITER)
        print("in the latest {} episodes:".format(10 * EPISODES_PER_ITER))
        print("black win count:", black_win_count)
        print("white win count:", white_win_count)
        print("draw count:", draw_count)
        print("run time: {} secs".format(round(time.time() - checkpoint_time)))

        checkpoint_time = time.time()
        black_win_count, white_win_count, draw_count = 0, 0, 0

print("total run time: {} secs".format(round(time.time() - start_time)))

if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
    print("\nblack model summary")
    dqn_black.model.summary()
    dqn_black.model.save(filepath='M1.h5')
elif (train_mode == '1' and model_choice == 'W') or train_mode == '2':
    print("\nwhite model summary")
    dqn_white.model.summary()
    dqn_white.model.save(filepath='M2.h5')
