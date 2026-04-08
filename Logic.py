import random
import gym
from gym_backgammon.envs.backgammon import Backgammon

def rollDice() -> tuple[int, int]:
    """Rolls two six-sided dice and returns the results as a tuple."""
    num1 = random.randint(1, 6)
    num2 = random.randint(1, 6)
    return (num1, num2)

class Board:
    def __init__(self, position_list=None):
        self.positions = [-2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0, 0, 0, 0] if position_list is None else position_list
        # index 0  to 23 represents number of checkers on points - negative for player one(dark), positive for player 2(light)
        # index 24 to 25 represents number of checkers on the bar - index 24 for P1, index 25 for P2
        # index 26 to 27 represents number of checkers off - index 26 for P1, index 27 for P2
        self.pip = (167,167) if position_list is None else self.calc_pips()
        self.bear_off_status = (False, False)
    
    def calc_pips(self) -> tuple[int, int]:
        """Calculates the total pips for both players."""
        p1_pip = 0
        p2_pip = 0
        for i in range(24):
            if self.positions[i] > 0:
                p2_pip += (i + 1) * self.positions[i]
            elif self.positions[i] < 0:
                p1_pip += (24 - i) * -self.positions[i]
        p1_pip += 25 * self.positions[24]
        p2_pip += 25 * self.positions[25]
        self.pip = (p1_pip, p2_pip)
        return self.pip

    def calc_bear_off_status(self):
        """Calculates the bear off status for both players."""
        p1_bear_off = True
        p2_bear_off = True
        for i in range(18):
            if self.positions[i] < 0:
                p1_bear_off = False
        for i in range(6, 24):
            if self.positions[i] > 0:
                p2_bear_off = False
        self.bear_off_status = (p1_bear_off, p2_bear_off)
        return self.bear_off_status

    def return_legal_moves(self, player: int, dice: tuple[int, int]) -> list:
        """Returns a list of legal moves for the given player and dice rolls."""
        current_player = 1 if player == 1 else 0
        if current_player == 0:
            dice = (-dice[0], -dice[1])

        game = Backgammon()
        game.board, game.bar, game.off = self._return_gym_transform()
        game.players_positions = game.get_players_positions()
        game.state = game.save_state()

        legal_moves = game.get_valid_plays(current_player,dice)
        return legal_moves
    
    def _return_gym_transform(self):
        """Returns a transformed version of the board for use in a gym environment."""
        
        X = [(0,None)] * 24
        for i in range(24):
            men = self.positions[i]
            if men > 0:
                X[i] = (abs(men),0)
            elif men < 0:
                X[i] = (abs(men),1)
        B = (self.positions[25], self.positions[24])
        O = (self.positions[27], self.positions[26])

        return X, B, O

    def _return_gym_transform_env(self, player: int) -> list:
        """Returns a transformed version of the board for use in a gym environment."""
        X = [0] * 198
        for i in range(24):
            men = self.positions[i]
            if men == 0:
                j1 = [0,0,0,0]
                j2 = [0,0,0,0]
            elif men == -1:
                j1 = [1,0,0,0]
                j2 = [0,0,0,0]
            elif men == 1:
                j1 = [0,0,0,0]
                j2 = [1,0,0,0]
            elif men == -2:
                j1 = [1,1,0,0]
                j2 = [0,0,0,0]
            elif men == 2:
                j1 = [0,0,0,0]
                j2 = [1,1,0,0]
            elif men <= -3:
                j1 = [1,1,1,((-men)-3)/2]
                j2 = [0,0,0,0]
            elif men >= 3:
                j1 = [0,0,0,0]
                j2 = [1,1,1,(men-3)/2]
            X[i:i+3] = j1
            X[i+98:i+101] = j2

        X[96] = self.positions[24]/2
        X[97] = self.positions[26]/15

        X[194] = self.positions[25]/2
        X[195] = self.positions[27]/15

        turn = [1,0] if player == 1 else [0,1]
        X[196:197] = turn

        return X


board = Board()
print(board.return_legal_moves(2, (2, 3)))