from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()  # Required for PyInstaller

    import os

    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (5, 35)

    def screen_clear():
        # for mac and linux(here, os.name is 'posix')
        if os.name == 'posix':
            _ = os.system('clear')
        else:
            # for windows platfrom
            _ = os.system('cls')
        # print out some text

    screen_clear()
    print("\nloading ...\n")

    import time
    import pygame
    import numpy as np
    from copy import deepcopy

    import minimax
    from board import Reversi
    import useful_func
    from tensorflow.keras.models import load_model


    class Screen_Contents:
        def __init__(self, screen, Rev):
            self.screen = screen
            self.Rev = Rev
            self.board_pos = (40, 110)

            self.points_pos = [(50, 30), (540, 30)]
            self.points_font = pygame.font.Font('data/font/Stanberry.ttf', 48)

            self.whowin_pos = (205, 30)
            self.whowin_font = pygame.font.Font("data/font/Stanberry.ttf", 48)

            self.whoturn_pos = (210, 30)
            self.whoturn_font = pygame.font.Font("data/font/Stanberry.ttf", 48)

            self.blank_board = pygame.image.load('data/img/blank_board560.png')
            self.board_frame = pygame.image.load('data/img/board_frame.png')
            self.w2b_sprites = [pygame.image.load("data/img/sprites/w2b/" + str(i) + ".png") for i in range(11)]
            self.b2w_sprites = [pygame.image.load("data/img/sprites/b2w/" + str(i) + ".png") for i in range(11)]

        def reset(self, screen, Rev):
            self.screen = screen
            self.Rev = Rev
            self.board_pos = (40, 110)

            self.points_pos = [(50, 30), (540, 30)]
            self.points_font = pygame.font.Font('data/font/Stanberry.ttf', 48)

            self.whowin_pos = (215, 30)
            self.whowin_font = pygame.font.Font("data/font/Stanberry.ttf", 48)

            self.whoturn_pos = (210, 30)
            self.whoturn_font = pygame.font.Font("data/font/Stanberry.ttf", 48)

        def board_sqr_surf(self):
            arr = pygame.surfarray.array3d(self.blank_board)
            squares = np.empty((8, 8, 70, 70, 3)).astype(int)
            squares_surf = []
            for i in range(8):
                for j in range(8):
                    squares[i][j] = arr[i * 70: (i + 1) * 70, j * 70: (j + 1) * 70, :]
            for j in range(8):
                for i in range(8):
                    squares_surf.append(pygame.surfarray.make_surface(squares[i][j]))
            return squares_surf

        def blit_board(self):
            self.screen.blit(sc.blank_board, (sc.board_pos[0], sc.board_pos[1]))
            filled_i, filled_j = np.where(sc.Rev.board != 2)
            filled = list(zip(filled_i, filled_j))
            for (i, j) in filled:
                if sc.Rev.board[i, j] == 0:
                    screen.blit(sc.b2w_sprites[0], (sc.board_pos[0] + 70 * j, sc.board_pos[1] - 10 + 70 * i))
                else:
                    screen.blit(sc.w2b_sprites[0], (sc.board_pos[0] + 70 * j, sc.board_pos[1] - 10 + 70 * i))

        def blit_points(self, temp_points):
            black_points, white_points = str(temp_points[0]), str(temp_points[1])
            if temp_points[0] < 10:
                black_points = '0' + black_points
            if temp_points[1] < 10:
                white_points = '0' + white_points
            points_surf = [self.points_font.render(black_points + '  ', True, (0, 0, 0), (100, 100, 100)),
                           self.points_font.render(white_points + '  ', True, (255, 255, 255), (100, 100, 100))]
            self.screen.blits([(points_surf[0], self.points_pos[0]), (points_surf[1], self.points_pos[1])])
            return [pygame.Rect(self.points_pos[k][0], self.points_pos[k][1], 96, 40) for k in range(2)]

        def blit_whowin(self):
            whowin = self.Rev.check_whowin()
            if whowin:
                whowin_surf = self.whowin_font.render(" White Won  ", True, (255, 255, 255), (100, 100, 100))
            elif whowin == 0:
                whowin_surf = self.whowin_font.render(" Black Won  ", True, (0, 0, 0), (100, 100, 100))
            else:
                whowin_surf = self.whowin_font.render("   Draw     ", True, (0, 200, 0), (100, 100, 100))
            sc.screen.blit(whowin_surf, self.whowin_pos)

        def blit_whoturn(self):
            if self.Rev.turn:
                whoturnsurf = self.whoturn_font.render("White Turn ", True, (255, 255, 255), (100, 100, 100))
            else:
                whoturnsurf = self.whoturn_font.render("Black Turn ", True, (0, 0, 0), (100, 100, 100))
            sc.screen.blit(whoturnsurf, self.whoturn_pos)

        def blit_squares(self, ij_list):
            squares_surf = self.board_sqr_surf()
            for (i, j) in ij_list:
                if i < 0:
                    screen.blit(sc.board_frame, (self.board_pos[0] - 18, self.board_pos[1] - 18))
                else:
                    self.screen.blit(squares_surf[i * 8 + j], (self.board_pos[0] + 70 * j, self.board_pos[1] + 70 * i))

        def put_piece(self, color, i, j):
            # self.blit_squares([(i, j)])
            # color is 1 if white, color is 0 if black
            last_sprite_index = len(self.b2w_sprites) - 1
            self.screen.blit(self.b2w_sprites[color * last_sprite_index],
                             (self.board_pos[0] + 70 * j, self.board_pos[1] - 10 + 70 * i))
            pygame.display.update(pygame.Rect(self.board_pos[0] + 70 * j, self.board_pos[1] + 70 * i, 70, 70))

        def flip_piece(self, i, j, temp_board):
            clock_flip = pygame.time.Clock()
            flip_FPS = 60
            if temp_board[i, j] == 1:  # if original color is white
                for f in range(1, len(self.w2b_sprites)):
                    clock_flip.tick(flip_FPS)
                    self.blit_squares([(i, j), (i - 1, j)])
                    self.screen.blit(self.w2b_sprites[f], (self.board_pos[0] + 70 * j, self.board_pos[1] - 10 + 70 * i))
                    if i > 0:
                        if temp_board[i - 1, j] != 2:
                            last_sprite_index = len(self.b2w_sprites) - 1
                            self.screen.blit(self.b2w_sprites[temp_board[i - 1, j] * last_sprite_index],
                                             (self.board_pos[0] + 70 * j, self.board_pos[1] - 10 + 70 * (i - 1)))
                    pygame.display.update(
                        [pygame.Rect(self.board_pos[0] + 70 * j, self.board_pos[1] + 70 * (i + k), 70, 70) for k in
                         range(-1, 1)])
            elif temp_board[i, j] == 0:
                for f in range(1, len(self.b2w_sprites)):
                    clock_flip.tick(flip_FPS)
                    self.blit_squares([(i, j), (i - 1, j)])
                    self.screen.blit(self.b2w_sprites[f], (self.board_pos[0] + 70 * j, self.board_pos[1] - 10 + 70 * i))
                    if i > 0:
                        if temp_board[i - 1, j] != 2:
                            last_sprite_index = len(self.b2w_sprites) - 1
                            self.screen.blit(self.b2w_sprites[temp_board[i - 1, j] * last_sprite_index],
                                             (self.board_pos[0] + 70 * j, self.board_pos[1] - 10 + 70 * (i - 1)))
                    pygame.display.update(
                        [pygame.Rect(self.board_pos[0] + 70 * j, self.board_pos[1] + 70 * (i + k), 70, 70) for k in
                         range(-1, 1)])


    def find_square(mloc):
        mx, my = mloc
        if sc.board_pos[0] <= mx < sc.board_pos[0] + 560 and sc.board_pos[1] <= my < sc.board_pos[1] + 560:
            j = (mx - sc.board_pos[0]) // 70
            i = (my - sc.board_pos[1]) // 70
            return (i, j)
        return None

    ### main game loop ###

    action_list = useful_func.create_actions_list()

    Rev = Reversi()

    mode_running = False
    main_running = True

    while main_running:
        screen_clear()

        print("\n\n  game-mode 1: human VS human")
        print("  game-mode 2: play against computer")
        while main_running:
            game_mode = input("\n  choose game-mode 1 or 2: ")
            if game_mode == 'exit':
                main_running = False
            elif game_mode == '1' or game_mode == '2':
                break
            else:
                print("  Invalid input, please try again, type '1' or '2' and then press enter")

        if game_mode == '1':
            mode_running = True

            screen_clear()

            Rev.reset()

            pygame.init()

            clock_main = pygame.time.Clock()

            screen_width = 560 + 40 * 2
            screen_height = 560 + 40 + 110
            screen = pygame.display.set_mode(size=(screen_width, screen_height))

            pygame.display.set_caption("Reversi (Human VS Human)")

            py_icon = pygame.image.load('data/img/py_icon.png')
            pygame.display.set_icon(py_icon)

            sc = Screen_Contents(screen, Rev)

            screen.fill((100, 100, 100))
            screen.blit(sc.blank_board, (sc.board_pos[0], sc.board_pos[1]))
            screen.blit(sc.board_frame, (sc.board_pos[0] - 18, sc.board_pos[1] - 18))
            pygame.display.update()

            print("\n\n  move 1 ...")

            while mode_running:
                clock_main.tick(20)

                sc.blit_board()

                sc.blit_points([Rev.n_black, Rev.n_white])

                if Rev.game_over():
                    sc.blit_whowin()
                else:
                    sc.blit_whoturn()

                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        del sc
                        pygame.quit()
                        mode_running = False

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if pygame.mouse.get_pressed(num_buttons=3) == (1, 0, 0):
                            mloc = pygame.mouse.get_pos()
                            move = find_square(mloc)
                            if move is not None:
                                i, j = move
                                if Rev.board[i, j] == 2:
                                    if Rev.check_legal(move, Rev.turn):
                                        temp_board = deepcopy(Rev.board)
                                        temp_points = deepcopy(
                                            [(temp_board == 0).sum(), (temp_board == 1).sum()])  # [n_black, n_white]
                                        move_color = deepcopy(Rev.turn)

                                        sc.put_piece(move_color, i, j)
                                        temp_points[int(move_color)] += 1
                                        pygame.display.update(sc.blit_points(temp_points))

                                        temp_board[i, j] = int(move_color)
                                        time.sleep(0.2)

                                        Rev.update_board(move, move_color)
                                        flip_pos_i, flip_pos_j = np.where(Rev.board != temp_board)
                                        flip_pos = list(zip(flip_pos_i, flip_pos_j))

                                        for (fi, fj) in flip_pos:
                                            sc.blit_points(temp_points)
                                            sc.flip_piece(fi, fj, temp_board)
                                            temp_points[0] += 1 - int(move_color) * 2
                                            temp_points[1] += int(move_color) * 2 - 1
                                            pygame.display.update(sc.blit_points(temp_points))
                                            temp_board[fi, fj] = int(not temp_board[fi, fj])
                                        del temp_board
                                        del temp_points
                                        del move_color

                                        if not Rev.game_over():
                                            print("\n\n  move {} ...".format(sum(Rev.board.flatten() != 2) - 3))
                                        else:
                                            print("\n\n  game ended in move {}".format(sum(Rev.board.flatten() != 2) - 4))

        elif game_mode == '2':
            mode_running = True

            print("\n\n  computer 1: Deep-Q-Network (DQN)")
            print("  computer 2: minimax")
            while mode_running:
                computer = input("\n  choose computer 1 or 2: ")
                if computer == 'exit':
                    mode_running = False
                    main_running = False
                if computer == '1' or computer == '2':
                    break
                else:
                    print("  Invalid input, please try again, type '1' or '2' and then press enter")

            if computer == '2':
                print("\n\n  available choices of depth of minimax: 1-6")
                print("  !!! higher depth requires longer computational time !!!")
                while mode_running:
                    depth = input("\n  choose depth (recommend: 2-4): ")
                    if depth == 'exit':
                        mode_running = False
                        main_running = False
                    elif len(depth) == 1 and depth in "123456":
                        depth = int(depth)
                        break
                    else:
                        print("  Invalid input, please try again, type any whole number between 1 to 6 inclusively and then press enter")
                if mode_running:
                    print("\n\n  !!! multiprocessing shortens computational time, but requires high cpu usage !!!")
                while mode_running:
                    multiproc = input("\n  apply multiprocessing (y/n): ")
                    if multiproc == 'exit':
                        mode_running = False
                        main_running = False
                    elif multiproc == 'y':
                        multiproc = True
                        break
                    elif multiproc == 'n':
                        multiproc = False
                        break
                    else:
                        print("  Invalid input, please try again, type 'y' or 'n' then press enter")

            human_side = None
            while mode_running:
                human_side = input("\n\n  you wanna play black or white? (B/W): ")
                if human_side == 'exit':
                    mode_running = False
                    main_running = False
                elif human_side == 'B':
                    human_side = False
                    break
                elif human_side == 'W':
                    human_side = True
                    break
                else:
                    print("  Invalid input, please try again, type 'B' or 'W' and then press enter")

            if not mode_running: continue
            ### mode 2 starts here ###
            if computer == '1':
                if human_side:
                    model = load_model('data/nn/M1.h5')
                else:
                    model = load_model('data/nn/M2.h5')
            elif computer == '2':
                # initialize minimax positions cache
                pos_cache = {}

            screen_clear()

            Rev.reset()

            pygame.init()

            clock_main = pygame.time.Clock()

            screen_width = 560 + 40 * 2
            screen_height = 560 + 40 + 110
            screen = pygame.display.set_mode(size=(screen_width, screen_height))

            if computer == '1':
                pygame.display.set_caption("Reversi (Play Against DQN)")
            elif computer == '2':
                pygame.display.set_caption("Reversi (Play Against Minimax)")
            py_icon = pygame.image.load('data/img/py_icon.png')
            pygame.display.set_icon(py_icon)

            sc = Screen_Contents(screen, Rev)

            screen.fill((100, 100, 100))
            screen.blit(sc.blank_board, (sc.board_pos[0], sc.board_pos[1]))
            screen.blit(sc.board_frame, (sc.board_pos[0] - 18, sc.board_pos[1] - 18))
            pygame.display.update()

            print("\n\n  move 1 ...")

            while mode_running:
                clock_main.tick(20)

                sc.blit_board()

                sc.blit_points([Rev.n_black, Rev.n_white])

                if Rev.game_over():
                    sc.blit_whowin()
                else:
                    sc.blit_whoturn()

                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        del sc
                        pygame.quit()
                        mode_running = False
                        if computer == '1':
                            del model
                        if computer == '2':
                            del pos_cache

                    if Rev.game_over(): continue

                    move = None
                    human_turn_label = (Rev.turn == human_side)
                    if human_turn_label:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if pygame.mouse.get_pressed(num_buttons=3) == (1, 0, 0):
                                mloc = pygame.mouse.get_pos()
                                move = find_square(mloc)
                                temp_Rev = deepcopy(Rev)
                                if move is not None:
                                    if Rev.check_legal(move, Rev.turn) and computer == '2':
                                        chosen_next_pos = minimax.new_position(Rev.board, Rev.neighbors, move, Rev.turn)
                                        chosen_next_pos_code = minimax.pos_to_code(chosen_next_pos)
                                        if chosen_next_pos_code in pos_cache.keys():
                                            pos_cache = pos_cache[chosen_next_pos_code]

                    else:
                        pygame.time.wait(1000)
                        legal_moves = Rev.legal_moves(Rev.turn)
                        if computer == '1':
                            x = useful_func.state_to_x(Rev.board)
                            y = model.predict(np.expand_dims(x, axis=0))
                            move = useful_func.ai_move(y, legal_moves, action_list)
                        elif computer == '2':
                            board_input = Rev.board
                            neighbors_input = Rev.neighbors
                            turn_input = Rev.turn

                            eval_start_time = time.time()
                            move, eval = minimax.minimax_move(not human_side, depth, legal_moves,
                                                              board_input, Rev.neighbors, turn_input, pos_cache, multiproc)
                            eval_time_elapsed = time.time() - eval_start_time

                            pygame.event.clear()
                            # update cache
                            chosen_next_pos = minimax.new_position(Rev.board, Rev.neighbors, move, Rev.turn)
                            chosen_next_pos_code = minimax.pos_to_code(chosen_next_pos)
                            if chosen_next_pos_code in pos_cache.keys():
                                pos_cache = pos_cache[chosen_next_pos_code]
                            # print move and eval

                            if eval > 0:
                                print("  computer's advantage evaluation score: +{}".format(eval))
                            else:
                                print("  computer's advantage evaluation score:", eval)
                            print("  evaluation time spent: {} ms".format(round(eval_time_elapsed * 1000)))

                    if move is not None:
                        i, j = move
                        if Rev.board[i, j] == 2:
                            if Rev.check_legal(move, Rev.turn):
                                temp_board = deepcopy(Rev.board)
                                temp_points = deepcopy(
                                    [(temp_board == 0).sum(), (temp_board == 1).sum()])  # [n_black, n_white]
                                move_color = deepcopy(Rev.turn)

                                sc.put_piece(move_color, i, j)
                                temp_points[int(move_color)] += 1
                                pygame.display.update(sc.blit_points(temp_points))

                                temp_board[i, j] = int(move_color)
                                time.sleep(0.2)

                                Rev.update_board(move, move_color)
                                flip_pos_i, flip_pos_j = np.where(Rev.board != temp_board)
                                flip_pos = list(zip(flip_pos_i, flip_pos_j))

                                for (fi, fj) in flip_pos:
                                    sc.blit_points(temp_points)
                                    sc.flip_piece(fi, fj, temp_board)
                                    temp_points[0] += 1 - int(move_color) * 2
                                    temp_points[1] += int(move_color) * 2 - 1
                                    pygame.display.update(sc.blit_points(temp_points))
                                    temp_board[fi, fj] = int(not temp_board[fi, fj])
                                del temp_board
                                del temp_points
                                del move_color

                                if not Rev.game_over():
                                    print("\n\n  move {} ...".format(sum(Rev.board.flatten() != 2) - 3))
                                else:
                                    print("\n\n  game ended in move {}".format(sum(Rev.board.flatten() != 2) - 4))

        while main_running:
            play_again = input("\n\n  play again? (y/n): ")
            if play_again == 'n':
                main_running = False
            elif play_again == 'y':
                break
            else:
                print("  type 'y' or 'n' and then press enter, try again")


    screen_clear()