## still in development
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
import numpy as np
import board
import time
import useful_func

Rev = board.Reversi()

# def shuffle_in_unison(self, a, b):
#     rng_state = np.random.get_state()
#     np.random.shuffle(a)
#     np.random.set_state(rng_state)
#     np.random.shuffle(b)

class PolicyNetwork(Model):
    def __init__(self, n_actions, hidden_layers_dims):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_layers_dims = hidden_layers_dims

    def call(self, X):
        hidden = Dense(units=self.hidden_layers_dims[0], activation='relu')(X)
        for i in range(1, len(self.hidden_layers_dims)):
            hidden = Dense(units=self.hidden_layers_dims[i], activation='relu')(hidden)
        pi = Dense(units=self.n_actions, activation='softmax')(hidden)
        return pi

class Agent:
    def __init__(self, n_actions, hidden_layers_dims, ALPHA, GAMMA, inherit = False, old_policy = None):
        self.n_actions = n_actions
        self.hidden_layers_dims = hidden_layers_dims
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.policy = old_policy
        if not inherit:
            self.policy = PolicyNetwork(n_actions, hidden_layers_dims)
            self.policy.compile(optimizer=Adam(learning_rate=ALPHA))
        self.memories = []
        self.loss_record = []

    def choose_action(self, x, possible_actions):
        # x should be a 1D array
        x_tensor = tf.convert_to_tensor([x], dtype=tf.float32)
        pdf = self.policy(x_tensor)
        actions_probs = tfp.distributions.Categorical(probs=pdf)
        while 1:
            action_tensor = actions_probs.sample()
            action = action_tensor.numpy()[0]
            if action in possible_actions:
                return action


    def train(self):
        with tf.GradientTape() as tape:
            loss = 0
            for memory in self.memories:
                actions = tf.convert_to_tensor([memory[idx][1] for idx in range(len(memory))], dtype=tf.float32)
                G = tf.convert_to_tensor([memory[idx][2] for idx in range(len(memory))], dtype=tf.float32)
                for idx, (x, _, _) in enumerate(memory):
                    x_tensor = tf.convert_to_tensor([x], dtype=tf.float32)
                    probs = self.policy(x_tensor)
                    actions_probs = tfp.distributions.Categorical(probs=probs)
                    log_probs = actions_probs.log_prob(actions[idx])
                    loss += -G[idx] * tf.squeeze(log_probs)  # log_probs have batch dimension (is 1D now),                                                 # squeeze function turn it to 0D
            gradient = tape.gradient(loss, self.policy.trainable_variables)

        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        self.loss_record.append(float(loss))


action_to_move = useful_func.create_actions_list()
def play_one_episode(train_mode, model_choice, agent_B, agent_W, GAMMA):
    Rev.reset()
    black_trajectory = []
    white_trajectory = []
    while 1:
        x = useful_func.state_to_x(Rev.board)
        possible_moves = Rev.legal_moves(Rev.turn)
        possible_actions = [action_to_move.index(move) for move in possible_moves]
        if Rev.turn:  # if white's turn
            if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
                action = agent_W.choose_action(x, possible_actions)
            else:
                action = None
        else:
            if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
                action = agent_B.choose_action(x, possible_actions)
            else:
                action = None

        if action is not None:
            move = action_to_move[action]
            if Rev.turn:
                white_trajectory.append([x, action])
            else:
                black_trajectory.append([x, action])
        else:
            move = useful_func.random_move(possible_moves)

        Rev.update_board(move, Rev.turn)

        if Rev.game_over():
            black_rewards = (1, -1, 0)
            white_rewards = (-1, 1, 0)
            whowin = Rev.check_whowin()
            black_reward = black_rewards[whowin]
            white_reward = white_rewards[whowin]

            if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
                for t in range(len(black_trajectory) - 1, -1, -1):
                    black_trajectory[t].append(black_reward)
                    black_reward = GAMMA * black_reward
            if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
                for t in range(len(white_trajectory) - 1, -1, -1):
                    white_trajectory[t].append(white_reward)
                    white_reward = GAMMA * white_reward
            # now the format of black_trajectory or white_trajectory: [[x, performed action, return], ...]
            return black_trajectory, white_trajectory, whowin

### main ###
ALPHA = 0.01
GAMMA = 0.95
NUM_ITERS = 20
# MEMORY_SIZE >= EPISODES_PER_ITER
EPISODES_PER_ITER = 1
MEMORY_SIZE = 1
SHOW_INTERVAL = 10
PRIOR_ITERS = 0

n_actions = 60
hidden_layers_dims = [1024, 1024, 1024]

print("\n  train mode 1: model VS random")
print("  train mode 2: model VS model")
while 1:
    train_mode = input("\n  select train mode: ")
    if train_mode == '1' or train_mode == '2':
        break
    else:
        print("  type 1 or 2 and then press enter, try again")

model_choice = None
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
            black_policy = load_model(r'./data/nn/pn1.h5')
        elif model_choice == 'W':
            white_policy = load_model(r'./data/nn/pn2.h5')
    else:
        black_policy = load_model(r'./data/nn/pn1.h5')
        white_policy = load_model(r'./data/nn/pn2.h5')
else:
    black_policy = None
    white_policy = None

if train_mode == '1':
    if model_choice == 'B':
        agent_black = Agent(n_actions, hidden_layers_dims, ALPHA, GAMMA, inherit, black_policy)
        agent_white = None
    elif model_choice == 'W':
        agent_white = Agent(n_actions, hidden_layers_dims, ALPHA, GAMMA, inherit, white_policy)
        agent_black = None
else:
    agent_black = Agent(n_actions, hidden_layers_dims, ALPHA, GAMMA, inherit, black_policy)
    agent_white = Agent(n_actions, hidden_layers_dims, ALPHA, GAMMA, inherit, white_policy)


win_count = [0, 0, 0]
loss_record = []

start_time = time.time()
checkpoint_time = time.time()
for iter in range(NUM_ITERS):
    for _ in range(EPISODES_PER_ITER):
        black_trajectory, white_trajectory, whowin = play_one_episode(train_mode, model_choice, agent_black, agent_white, GAMMA)
        win_count[whowin] += 1
        # train model if replay buffer len > certain number
        if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
            if len(agent_black.memories) >= MEMORY_SIZE:
                # remove the oldest episode in replay buffer
                agent_black.memories.pop(0)
            # add the current_episode
            agent_black.memories.append(black_trajectory)
        if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
            if len(agent_white.memories) >= MEMORY_SIZE:
                # remove the oldest episode in replay buffer
                agent_white.memories.pop(0)
            # add the current_episode
            agent_white.memories.append(white_trajectory)
    if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
        if len(agent_black.memories) == MEMORY_SIZE:
            agent_black.train()
    if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
        if len(agent_white.memories) == MEMORY_SIZE:
            agent_white.train()

    # show_loss = (iter % SHOW_INTERVAL == 0)
    # if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
    #     if len(black_replay_buffer) == REPLAY_BUFFER_SIZE:
    #         dqn_black.train(black_replay_buffer, show_loss)
    # if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
    #     if len(white_replay_buffer) == REPLAY_BUFFER_SIZE:
    #         dqn_white.train(white_replay_buffer, show_loss)

### presenting the win count and iterations passed ###
    if iter % SHOW_INTERVAL == SHOW_INTERVAL - 1:
        print("\nepisodes played:", (iter + 1) * EPISODES_PER_ITER)
        print("in the latest {} episodes:".format(10 * EPISODES_PER_ITER))
        print("black win count:", win_count[0])
        print("white win count:", win_count[1])
        print("draw count:", win_count[2])
        if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
            # print("black_record len:", len(agent_black.loss_record))
            print("avg. agent_black loss:", sum(agent_black.loss_record[-SHOW_INTERVAL:])/SHOW_INTERVAL)
            agent_black.loss_record = []
        if (train_mode == '1' and model_choice == 'W') or train_mode == '2':
            # print("white_record len:", len(agent_white.loss_record))
            print("agent_white loss:", sum(agent_white.loss_record[-SHOW_INTERVAL:])/SHOW_INTERVAL)
            agent_white.loss_record = []
        print("run time: {} secs".format(round(time.time() - checkpoint_time)))

        checkpoint_time = time.time()
        win_count = [0, 0, 0]

print("total run time: {} secs".format(round(time.time() - start_time)))

if (train_mode == '1' and model_choice == 'B') or train_mode == '2':
    print("\nblack model summary")
    agent_black.policy.summary()
    # agent_black.policy.save(filepath=r'./data/nn/pn1.h5')
elif (train_mode == '1' and model_choice == 'W') or train_mode == '2':
    print("\nwhite model summary")
    agent_white.policy.summary()
    # agent_white.policy.save(filepath=r'./data/nn/pn2.h5')