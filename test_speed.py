from board import Reversi
import useful_func
import time

Rev = Reversi()
Rev.reset()

n_games = 1000

win_count = [0, 0, 0]
start_time = time.time()
for _ in range(n_games):
    Rev.reset()
    while 1:
        if Rev.game_over():
            whowin = Rev.check_whowin()
            win_count[whowin] += 1
            # print("whowin:", whowin)
            # print("points difference:", Rev.n_black - Rev.n_white)
            break

        possible_moves = Rev.legal_moves(Rev.turn)
        move = useful_func.random_move(possible_moves)
        Rev.update_board(move, Rev.turn)

total_time = time.time() - start_time
print("total time:", total_time)
print("avg time:", total_time / n_games)
print(win_count[0])
print(win_count[1])
print(win_count[2])