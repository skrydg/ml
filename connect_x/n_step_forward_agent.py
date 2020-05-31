import numpy as np

INF = 1e9
BOARD_SHAPE = (6, 7)
FIRST_LABLE = 1
SECOND_LABLE = 2
EMPTY_LABLE = 0

def in_range(l, s, r):
    return l <= s < r

class NStepForwardAgent():
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    WIN_SCORE = 1e8

    def __init__(self, num_steps):
        self.steps = num_steps

    def get_line_score(self, board, start, d, lable):
        cnt_our = 0
        for i in range(4):
            if (board[start[0], start[1]] == lable):
                cnt_our += 1
            else:
                if (board[start[0], start[1]] != EMPTY_LABLE):
                    return 0
            start = (start[0] + d[0], start[1] + d[1])
        if (cnt_our == 4):
            return INF
        return cnt_our * cnt_our

    def get_score(self, b):
        score = 0
        for i in range(BOARD_SHAPE[0]):
            for j in range(BOARD_SHAPE[1]):
                for d in self.directions:
                    last_i = i + d[0] * 3
                    last_j = j + d[1] * 3
                    if in_range(0, last_i, BOARD_SHAPE[0]) and in_range(0, last_j, BOARD_SHAPE[1]):
                        score += self.get_line_score(b, (i, j), d, FIRST_LABLE) - self.get_line_score(b, (i, j), d,
                                                                                                      SECOND_LABLE)
        return score

    def get_move(self, board, i):
        j = BOARD_SHAPE[0] - 1
        while (j >= 0 and board[j, i] != 0):
            j -= 1
        assert (j >= 0)
        return (j, i)

    def brute_force(self, board, num_steps, lable, should_max):
        if self.get_score(board) > self.WIN_SCORE:
            return None, INF

        if self.get_score(board) < -self.WIN_SCORE:
            return None, -INF

        if num_steps == 0:
            return None, self.get_score(board)

        moves_score = [0] * BOARD_SHAPE[1]
        for i in range(BOARD_SHAPE[1]):
            if board[0, i] == EMPTY_LABLE:
                cur_move = self.get_move(board, i)
                board[cur_move[0], cur_move[1]] = lable
                _, moves_score[i] = self.brute_force(
                    board,
                    num_steps - 1,
                    FIRST_LABLE + SECOND_LABLE - lable,
                    not should_max
                )
                board[cur_move[0], cur_move[1]] = EMPTY_LABLE
            else:
                moves_score[i] = -INF if should_max else INF
        if should_max:
            best_move = np.argmax(moves_score)
        else:
            best_move = np.argmin(moves_score)

        return (best_move, moves_score[best_move])

    def act(self, observation, configuration):
        lable = observation['mark']
        should_max = (lable == FIRST_LABLE)
        b = np.array(observation['board']).reshape(BOARD_SHAPE)

        move, score = self.brute_force(b, self.steps, lable, should_max)

        return int(move)

agent = NStepForwardAgent(2)
def act(observation, configuration):
    return agent.act(observation, configuration)