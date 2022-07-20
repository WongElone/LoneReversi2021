from pygame import draw, image

class Screen_Contents:
    def __init__(self):
        pass

    def blank_board(self):
        blank_board = image.load('img/560x560green.png')
        draw.rect(blank_board, color=(20, 40, 20), rect=(0, 0, 1, 560))
        draw.rect(blank_board, color=(20, 40, 20), rect=(0, 0, 560, 1))
        for i in range(1,8):
            draw.rect(blank_board, color=(20, 40, 20), rect=(70 * i - 1, 0, 2, 560))
            draw.rect(blank_board, color=(20, 40, 20), rect=(0, 70 * i - 1, 560, 2))
        draw.rect(blank_board, color=(20, 40, 20), rect=(559, 0, 1, 560))
        draw.rect(blank_board, color=(20, 40, 20), rect=(0, 559, 560, 1))

        draw.circle(blank_board, color=(20, 40, 20), center=(140, 140), radius=5)
        draw.circle(blank_board, color=(20, 40, 20), center=(420, 140), radius=5)
        draw.circle(blank_board, color=(20, 40, 20), center=(140, 420), radius=5)
        draw.circle(blank_board, color=(20, 40, 20), center=(420, 420), radius=5)
        return blank_board


sc = Screen_Contents()
image.save(sc.blank_board(), 'img/test.png')