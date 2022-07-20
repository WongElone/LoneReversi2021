import numpy as np

class Reversi:
    # 0 is black, 1 is white, 2 is empty
    # start_board = np.zeros((8, 8)).astype(int) + 2
    # start_board[3, 3], start_board[4, 4], start_board[3, 4], start_board[4, 3] = 0, 0, 1, 1
    def __init__(self):
        self.board = np.zeros((8, 8)).astype(int) + 2
        self.board[3, 3], self.board[4, 4], self.board[3, 4], self.board[4, 3] = 1, 1, 0, 0
        self.turn = False    # False is bool(0) which is black's turn, vice versa, note that black plays first
        self.n_black = 2
        self.n_white = 2
        self.neighbors = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)}

    def check_legal(self, move, myside):
        # this methods check if a move is legal for any side
        # move is a tuple (i, j) where i is the row index, j is the col index
        # myside is a boolean indicating which side you wanna check
        i, j = move
        if (i, j) not in self.neighbors:
            return False
        if self.board[i, j] != 2:
            return False
        # check Up
        if i > 1:    # requirement of Up legal
            Up = np.flip(self.board[:i, j].flatten())  # need to flip because we want the first element to be the closest one
            if len(np.where(Up == int(myside))[0]) != 0:  # check if there exist same color
                distance = np.where(Up == int(myside))[0][0]  # close_same is the distance of closest same color
                if distance > 0:  # check if it is not adjacent
                    if set(Up[0:distance]) == {int(not myside)}: return True  # check if only opposite color exist before closest same color
        # check Down
        if i < 6:
            Down = self.board[i + 1:, j].flatten()
            if len(np.where(Down == int(myside))[0]) != 0:
                distance = np.where(Down == int(myside))[0][0]
                if distance > 0:
                    if set(Down[0:distance]) == {int(not myside)}: return True
        # check Left
        if j > 1:
            Left = np.flip(self.board[i, : j])
            if len(np.where(Left == int(myside))[0]) != 0:
                distance = np.where(Left == int(myside))[0][0]
                if distance > 0:
                    if set(Left[0:distance]) == {int(not myside)}: return True
        # check Right
        if j < 6:
            Right = self.board[i, j + 1:]
            if len(np.where(Right == int(myside))[0]) != 0:
                distance = np.where(Right == int(myside))[0][0]
                if distance > 0:
                    if set(Right[0:distance]) == {int(not myside)}: return True
        # check UpLeft
        if i > 1 and j > 1:
            size = min(i, j)
            square = self.board[(i - size): i, (j - size): j]
            UpLeft = np.flip(square.diagonal())
            if len(np.where(UpLeft == int(myside))[0]) != 0:
                distance = np.where(UpLeft == int(myside))[0][0]
                if distance > 0:
                    if set(UpLeft[0:distance]) == {int(not myside)}: return True
        # check UpRight
        if i > 1 and j < 6:
            size = min(i, 7 - j)
            square = self.board[(i - size): i, (j + 1): (j + 1 + size)]
            UpRight = np.flipud(square).diagonal()
            if len(np.where(UpRight == int(myside))[0]) != 0:
                distance = np.where(UpRight == int(myside))[0][0]
                if distance > 0:
                    if set(UpRight[0:distance]) == {int(not myside)}: return True
        # check DownLeft
        if i < 6 and j > 1:
            size = min(7 - i, j)
            square = self.board[(i + 1): (i + 1 + size), (j - size):j]
            DownLeft = np.flip(np.flipud(square).diagonal())
            if len(np.where(DownLeft == int(myside))[0]) != 0:
                distance = np.where(DownLeft == int(myside))[0][0]
                if distance > 0:
                    if set(DownLeft[0:distance]) == {int(not myside)}: return True
        # check DownRight
        if i < 6 and j < 6:
            size = min(7 - i, 7 - j)
            square = self.board[(i + 1): (i + 1 + size), (j + 1): (j + 1 + size)]
            DownRight = square.diagonal()
            if len(np.where(DownRight == int(myside))[0]) != 0:
                distance = np.where(DownRight == int(myside))[0][0]
                if distance > 0:
                    if set(DownRight[0:distance]) == {int(not myside)}: return True
        # not satisfying all above legal conditions
        return False

    def legal_moves(self, side):    # return all legal moves for any side
        LM = []
        for move in self.neighbors:
            if self.check_legal(move, side): LM.append(move)
        return LM

    def next_turn(self, current_turn):    # tells which side plays next turn, given current turn
        LM = self.legal_moves(not current_turn)
        if len(LM) == 0: return current_turn
        return not current_turn

    def update_neighbors(self, move):
        i, j = move
        new_neighbors = []
        for ii in range(i - 1, i + 2):
            if ii < 0 or ii > 7: continue    # this prevent indexing outside the board
            for jj in range(j - 1, j + 2):
                if jj < 0 or jj > 7: continue    # this prevent indexing outside the board
                if (ii, jj) == (i, j): continue    # this ignore the square just occupied by the latest move, since it should be removed from neighbors
                if self.board[ii, jj] != 2: continue    # filter out the squares already occupied
                new_neighbors.append((ii, jj))
        if len(new_neighbors) > 0:
            self.neighbors.update(new_neighbors)
        self.neighbors.remove((i, j))    # this remove the square just occupied by the latest move

    def update_board(self, move, myside):
        # this method updates the board, neighbors, points, and turn
        i, j = move
        # update center
        self.board[i, j] = int(myside)
        # update Up
        if i > 1:    # requirement of Up legal
            Up = np.flip(self.board[:i, j].flatten())    # need to flip because we want the first element to be the closest one
            if len(np.where(Up == int(myside))[0]) != 0:    # check if there exist same color
                distance = np.where(Up == int(myside))[0][0]    # close_same is the distance of closest same color
                if distance > 0:  # check if it is not adjacent
                    if set(Up[0:distance]) == {int(not myside)}:    # check if only opposite color exist before closest same color
                        self.board[(i - distance) : i, j] = int(myside)    # update color, tips: distance somehow indicates the number of pieces to change color
        # update Down
        if i < 6:
            Down = self.board[i + 1:, j].flatten()
            if len(np.where(Down == int(myside))[0]) != 0:
                distance = np.where(Down == int(myside))[0][0]
                if distance > 0:
                    if set(Down[0:distance]) == {int(not myside)}:
                        self.board[(i + 1) : (i + 1 + distance), j] = int(myside)
        # update Left
        if j > 1:
            Left = np.flip(self.board[i, : j])
            if len(np.where(Left == int(myside))[0]) != 0:
                distance = np.where(Left == int(myside))[0][0]
                if distance > 0:
                    if set(Left[0:distance]) == {int(not myside)}:
                        self.board[i, (j - distance) : j] = int(myside)
        # update Right
        if j < 6:
            Right = self.board[i, j + 1:]
            if len(np.where(Right == int(myside))[0]) != 0:
                distance = np.where(Right == int(myside))[0][0]
                if distance > 0:
                    if set(Right[0:distance]) == {int(not myside)}:
                        self.board[i, (j + 1) : (j + 1 + distance)] = int(myside)
        # update UpLeft
        if i > 1 and j > 1:
            size = min(i, j)
            square = self.board[(i - size): i, (j - size): j]
            UpLeft = np.flip(square.diagonal())
            if len(np.where(UpLeft == int(myside))[0]) != 0:
                distance = np.where(UpLeft == int(myside))[0][0]
                if distance > 0:
                    if set(UpLeft[0:distance]) == {int(not myside)}:
                        for k in range(distance):
                            self.board[i - 1 - k, j - 1 - k] = int(myside)
        # update UpRight
        if i > 1 and j < 6:
            size = min(i, 7 - j)
            square = self.board[(i - size): i, (j + 1): (j + 1 + size)]
            UpRight = np.flipud(square).diagonal()
            if len(np.where(UpRight == int(myside))[0]) != 0:
                distance = np.where(UpRight == int(myside))[0][0]
                if distance > 0:
                    if set(UpRight[0:distance]) == {int(not myside)}:
                        for k in range(distance):
                            self.board[i - 1 - k, j + 1 + k] = int(myside)
        # update DownLeft
        if i < 6 and j > 1:
            size = min(7 - i, j)
            square = self.board[(i + 1): (i + 1 + size), (j - size):j]
            DownLeft = np.flip(np.flipud(square).diagonal())
            if len(np.where(DownLeft == int(myside))[0]) != 0:
                distance = np.where(DownLeft == int(myside))[0][0]
                if distance > 0:
                    if set(DownLeft[0:distance]) == {int(not myside)}:
                        for k in range(distance):
                            self.board[i + 1 + k, j - 1 - k] = int(myside)
        # update DownRight
        if i < 6 and j < 6:
            size = min(7 - i, 7 - j)
            square = self.board[(i + 1): (i + 1 + size), (j + 1): (j + 1 + size)]
            DownRight = square.diagonal()
            if len(np.where(DownRight == int(myside))[0]) != 0:
                distance = np.where(DownRight == int(myside))[0][0]
                if distance > 0:
                    if set(DownRight[0:distance]) == {int(not myside)}:
                        for k in range(distance):
                            self.board[i + 1 + k, j + 1 + k] = int(myside)

        # update neighbors
        self.update_neighbors(move)
        # update points
        self.n_black = len(np.where(self.board == 0)[0])
        self.n_white = len(np.where(self.board == 1)[0])
        # update turn
        self.turn = self.next_turn(self.turn)

    def game_over(self):
        for move in self.neighbors:
            if self.check_legal(move, False): return False    # False is the black side
            if self.check_legal(move, True): return False    # True is the white side
        return True

    def check_whowin(self):
        if self.game_over():
            if self.n_black > self.n_white: return 0
            if self.n_black < self.n_white: return 1
            return 2

    def reset(self):
        self.board = np.zeros((8, 8)).astype(int) + 2
        self.board[3, 3], self.board[4, 4], self.board[3, 4], self.board[4, 3] = 1, 1, 0, 0
        self.turn = False  # False is bool(0) which is black's turn, vice versa, note that black plays first
        self.n_black = 2
        self.n_white = 2
        self.neighbors = {(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)}
