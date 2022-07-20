# play reversi with the console as the game interface

import os
import time
import numpy as np
from copy import deepcopy
from board import Reversi

Rev = Reversi()

def print_rules(board):
    symbol = ['⚪', '⚫', '  ']

    stopper = True
    print("\n      Please make sure you run this program on the command prompt in Windows,")
    print("      or any consoles in other operating systems.\n")
    print("      Otherwise, screen refresh and character-spacings might run into problems.\n")
    time.sleep(3)
    stopper = False
    if not stopper:
        _ = input("      press enter to continue... ")
    screen_clear()

    print("                  ->->->->->->")
    print("                  ¦¦ REVERSI ¦¦")
    print("                  <-<-<-<-<-<-")
    print("\n\n\n")
    print("\n      a    b    c    d    e    f    g    h", end="")
    for i in range(8):
        print("\n    ----------------------------------------\n  %d ¦" % (i + 1), end="")
        for j in range(8):
            print("    ¦", end="")
        print(" %d" % (i + 1), end="")
    print("\n    ----------------------------------------\n      a    b    c    d    e    f    g    h")

    time.sleep(1.5)
    print("\n    Instructions:\n")
    print("    1. This is a player-VS-player game-mode.\n")
    print("    2. The coordinates of the squares on the board are indicated by 'a' to 'h' and 1 to 8.\n")
    print("       For example, 'a1' indicates the square at the upper left corner.\n")
    print("    3. Type the coordinate of the move (e.g. 'a1') and then press enter to input your move\n")
    print("    4. If you have any troubles finding possible moves, you can type 'help' at anytime and press enter.")
    print("       All possible moves will then be shown on the board.\n")
    print("    5. You can type 'exit' and press enter to exit the game at anytime.\n")
    time.sleep(3)

def print_board(board, n_black, n_white, turn):
    symbol = ['⚪', '⚫', '  ']
    print("                  ->->->->->->")
    print("                  ¦¦ REVERSI ¦¦")
    print("                  <-<-<-<-<-<-")
    print("\n    Player %s:  %d" % (symbol[0], n_black), end="              ")
    print("Player %s:  %d" % (symbol[1], n_white))
    print("\n                 Player %s Turn" % symbol[int(turn)])
    print("\n      a    b    c    d    e    f    g    h", end="")
    for i in range(8):
        print("\n    ----------------------------------------\n  %d ¦" % (i + 1), end="")
        for j in range(8):
            print(" %s ¦" % symbol[board[i, j]], end="")
        print(" %d" % (i + 1), end="")
    print("\n    ----------------------------------------\n      a    b    c    d    e    f    g    h")

def print_board_show_move(board, n_black, n_white, turn):
    board_show_move = deepcopy(board)
    for (i, j) in Rev.legal_moves(turn): board_show_move[i, j] = 3    # use 3 to indicate neighbors

    symbol = ['⚪', '⚫', '  ', '. ']
    print("\n                  ->->->->->->")
    print("                  ¦¦ REVERSI ¦¦")
    print("                  <-<-<-<-<-<-")
    print("\n    Player %s:  %d" % (symbol[0], n_black), end="              ")
    print("Player %s:  %d" % (symbol[1], n_white))
    print("\n                 Player %s Turn" % symbol[int(turn)])
    print("\n      a    b    c    d    e    f    g    h", end="")
    for i in range(8):
        print("\n    ----------------------------------------\n  %d ¦" % (i + 1), end="")
        for j in range(8):
            print(" %s ¦" % symbol[board_show_move[i, j]], end="")
        print(" %d" % (i + 1), end="")
    print("\n    ----------------------------------------\n      a    b    c    d    e    f    g    h")

def print_board_over(board, n_black, n_white):
    symbol = ['⚪', '⚫', '  ']
    print("                  ->->->->->->")
    print("                  ¦¦ REVERSI ¦¦")
    print("                  <-<-<-<-<-<-")
    print("\n    Player %s:  %d" % (symbol[0], n_black), end="              ")
    print("Player %s:  %d" % (symbol[1], n_white))
    if n_black == n_white:
        print("\n                      Draw")
    else:
        print("\n                 Player %s Won" % symbol[np.argmax([n_black, n_white])])
    print("\n      a    b    c    d    e    f    g    h", end="")
    for i in range(8):
        print("\n    ----------------------------------------\n  %d ¦" % (i + 1), end="")
        for j in range(8):
            print(" %s ¦" % symbol[board[i, j]], end="")
        print(" %d" % (i + 1), end="")
    print("\n    ----------------------------------------\n      a    b    c    d    e    f    g    h")
    print("\n    Game Over")


def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')
   # print out some text

def ask_move():
    poss_moves = {'a':'1', 'b':'2', 'c':'3', 'd':'4', 'e':'5', 'f':'6', 'g':'7', 'h':'8'}
    symbol = ['⚪', '⚫']
    # check if the input follows the dictionary
    # check if the move is legal
    screen_clear()
    print_board(Rev.board, Rev.n_black, Rev.n_white, Rev.turn)
    while True:
        time.sleep(1)
        move = input("\n    next move: ")
        time.sleep(0.5)
        screen_clear()
        print_board(Rev.board, Rev.n_black, Rev.n_white, Rev.turn)
        if move == "exit":
            return False
        if move == "help":
            screen_clear()
            print_board_show_move(Rev.board, Rev.n_black, Rev.n_white, Rev.turn)
        elif len(move) != 2:
            print("\n    Warning: Invalid Input")
        elif move[0] in poss_moves.keys() and move[1] in poss_moves.values():
            i, j = int(move[1]) - 1, int(poss_moves[move[0]]) - 1
            if Rev.check_legal(move=(i, j), myside=Rev.turn):
                return (i, j)
            print("\n    Warning: Impossible Move")
            time.sleep(0.5)
            print("    If you have trouble finding possible moves, type 'help' to show all possible moves")
        else:
            print("\n    Warning: Invalid Input")


### Main Game ###
Rev.reset()

screen_clear()
print_rules(Rev.board)

exit = input("    press enter to continue... ")
if exit != 'exit':
    while True:
        screen_clear()
        print_board(Rev.board, Rev.n_black, Rev.n_white, Rev.turn)

        if Rev.game_over():
            screen_clear()
            print_board_over(Rev.board, Rev.n_black, Rev.n_white)
            again = input("\n    one more game? (y/n): ")
            if again == 'y' or again == 'Y' or again == 'yes' or again == 'Yes':
                Rev.reset()
                continue
            break

        move = ask_move()  # ask_move() already check if it is legal move
        if move is False:
            break
        Rev.update_board(move=move, myside=Rev.turn)


