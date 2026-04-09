import random

from gym_backgammon.envs.backgammon import Backgammon # pyright: ignore[reportMissingImports]

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
        return list(legal_moves)
    
    def execute_move(self, player: int, move: list):
        """Executes the given move for the given player."""
        for move_part in move:
            src, dest = move_part

            if src == 'bar':
                from_index = 24 if player == 1 else 25
                self.positions[from_index] -= 1
            else:
                from_index = src
                self.positions[from_index] += 1 if player == 1 else -1

            if dest in ['off', 24, -1]:
                to_index = 26 if player == 1 else 27
                self.positions[to_index] += 1
            else:
                to_index = dest
                if player == 1:
                    if self.positions[to_index] == 1:
                        self.positions[to_index] = -1
                        self.positions[25] += 1
                    else:
                        self.positions[to_index] -= 1
                else:
                    if self.positions[to_index] == -1:
                        self.positions[to_index] = 1
                        self.positions[24] += 1
                    else:
                        self.positions[to_index] += 1

    def render_terminal(self):
        positions = self.positions

        a1 = "  O " if positions[0] > 0 else "  X " if positions[0] < 0 else "    "
        a2 = "  O " if positions[1] > 0 else "  X " if positions[1] < 0 else "    "
        a3 = "  O " if positions[2] > 0 else "  X " if positions[2] < 0 else "    "
        a4 = "  O " if positions[3] > 0 else "  X " if positions[3] < 0 else "    "
        a5 = "  O " if positions[4] > 0 else "  X " if positions[4] < 0 else "    "
        a6 = "  O " if positions[5] > 0 else "  X " if positions[5] < 0 else "    "
        a7 = "  O " if positions[6] > 0 else "  X " if positions[6] < 0 else "    "
        a8 = "  O " if positions[7] > 0 else "  X " if positions[7] < 0 else "    "
        a9 = "  O " if positions[8] > 0 else "  X " if positions[8] < 0 else "    "
        aa = "  O " if positions[9] > 0 else "  X " if positions[9] < 0 else "    "
        ab = "  O " if positions[10]> 0 else "  X " if positions[10]< 0 else "    "
        ac = "  O " if positions[11]> 0 else "  X " if positions[11]< 0 else "    "
        b1 = "  O " if positions[0] > 1 else "  X " if positions[0] < -1 else "    "
        b2 = "  O " if positions[1] > 1 else "  X " if positions[1] < -1 else "    "
        b3 = "  O " if positions[2] > 1 else "  X " if positions[2] < -1 else "    "
        b4 = "  O " if positions[3] > 1 else "  X " if positions[3] < -1 else "    "
        b5 = "  O " if positions[4] > 1 else "  X " if positions[4] < -1 else "    "
        b6 = "  O " if positions[5] > 1 else "  X " if positions[5] < -1 else "    "
        b7 = "  O " if positions[6] > 1 else "  X " if positions[6] < -1 else "    "
        b8 = "  O " if positions[7] > 1 else "  X " if positions[7] < -1 else "    "
        b9 = "  O " if positions[8] > 1 else "  X " if positions[8] < -1 else "    "
        ba = "  O " if positions[9] > 1 else "  X " if positions[9] < -1 else "    "
        bb = "  O " if positions[10]> 1 else "  X " if positions[10]< -1 else "    "
        bc = "  O " if positions[11]> 1 else "  X " if positions[11]< -1 else "    "
        c1 = "  O " if positions[0] > 2 else "  X " if positions[0] < -2 else "    "
        c2 = "  O " if positions[1] > 2 else "  X " if positions[1] < -2 else "    "
        c3 = "  O " if positions[2] > 2 else "  X " if positions[2] < -2 else "    "
        c4 = "  O " if positions[3] > 2 else "  X " if positions[3] < -2 else "    "
        c5 = "  O " if positions[4] > 2 else "  X " if positions[4] < -2 else "    "
        c6 = "  O " if positions[5] > 2 else "  X " if positions[5] < -2 else "    "
        c7 = "  O " if positions[6] > 2 else "  X " if positions[6] < -2 else "    "
        c8 = "  O " if positions[7] > 2 else "  X " if positions[7] < -2 else "    "
        c9 = "  O " if positions[8] > 2 else "  X " if positions[8] < -2 else "    "
        ca = "  O " if positions[9] > 2 else "  X " if positions[9] < -2 else "    "
        cb = "  O " if positions[10]> 2 else "  X " if positions[10]< -2 else "    "
        cc = "  O " if positions[11]> 2 else "  X " if positions[11]< -2 else "    "
        d1 = "  O " if positions[0] > 3 else "  X " if positions[0] < -3 else "    "
        d2 = "  O " if positions[1] > 3 else "  X " if positions[1] < -3 else "    "
        d3 = "  O " if positions[2] > 3 else "  X " if positions[2] < -3 else "    "
        d4 = "  O " if positions[3] > 3 else "  X " if positions[3] < -3 else "    "
        d5 = "  O " if positions[4] > 3 else "  X " if positions[4] < -3 else "    "
        d6 = "  O " if positions[5] > 3 else "  X " if positions[5] < -3 else "    "
        d7 = "  O " if positions[6] > 3 else "  X " if positions[6] < -3 else "    "
        d8 = "  O " if positions[7] > 3 else "  X " if positions[7] < -3 else "    "
        d9 = "  O " if positions[8] > 3 else "  X " if positions[8] < -3 else "    "
        da = "  O " if positions[9] > 3 else "  X " if positions[9] < -3 else "    "
        db = "  O " if positions[10]> 3 else "  X " if positions[10]< -3 else "    "
        dc = "  O " if positions[11]> 3 else "  X " if positions[11]< -3 else "    "
        e1 = "  O " if positions[0] > 4 else "  X " if positions[0] < -4 else "    "
        e2 = "  O " if positions[1] > 4 else "  X " if positions[1] < -4 else "    "
        e3 = "  O " if positions[2] > 4 else "  X " if positions[2] < -4 else "    "
        e4 = "  O " if positions[3] > 4 else "  X " if positions[3] < -4 else "    "
        e5 = "  O " if positions[4] > 4 else "  X " if positions[4] < -4 else "    "
        e6 = "  O " if positions[5] > 4 else "  X " if positions[5] < -4 else "    "
        e7 = "  O " if positions[6] > 4 else "  X " if positions[6] < -4 else "    "
        e8 = "  O " if positions[7] > 4 else "  X " if positions[7] < -4 else "    "
        e9 = "  O " if positions[8] > 4 else "  X " if positions[8] < -4 else "    "
        ea = "  O " if positions[9] > 4 else "  X " if positions[9] < -4 else "    "
        eb = "  O " if positions[10]> 4 else "  X " if positions[10]< -4 else "    "
        ec = "  O " if positions[11]> 4 else "  X " if positions[11]< -4 else "    "
        f1 = "  + " if positions[0] > 5 else "  + " if positions[0] < -5 else "    "
        f2 = "  + " if positions[1] > 5 else "  + " if positions[1] < -5 else "    "
        f3 = "  + " if positions[2] > 5 else "  + " if positions[2] < -5 else "    "
        f4 = "  + " if positions[3] > 5 else "  + " if positions[3] < -5 else "    "
        f5 = "  + " if positions[4] > 5 else "  + " if positions[4] < -5 else "    "
        f6 = "  + " if positions[5] > 5 else "  + " if positions[5] < -5 else "    "
        f7 = "  + " if positions[6] > 5 else "  + " if positions[6] < -5 else "    "
        f8 = "  + " if positions[7] > 5 else "  + " if positions[7] < -5 else "    "
        f9 = "  + " if positions[8] > 5 else "  + " if positions[8] < -5 else "    "
        fa = "  + " if positions[9] > 5 else "  + " if positions[9] < -5 else "    "
        fb = "  + " if positions[10]> 5 else "  + " if positions[10]< -5 else "    "
        fc = "  + " if positions[11]> 5 else "  + " if positions[11]< -5 else "    "

        g1 = "  + " if positions[23] > 5 else "  + " if positions[23] < -5 else "    "
        g2 = "  + " if positions[22] > 5 else "  + " if positions[22] < -5 else "    "
        g3 = "  + " if positions[21] > 5 else "  + " if positions[21] < -5 else "    "
        g4 = "  + " if positions[20] > 5 else "  + " if positions[20] < -5 else "    "
        g5 = "  + " if positions[19] > 5 else "  + " if positions[19] < -5 else "    "
        g6 = "  + " if positions[18] > 5 else "  + " if positions[18] < -5 else "    "
        g7 = "  + " if positions[17] > 5 else "  + " if positions[17] < -5 else "    "
        g8 = "  + " if positions[16] > 5 else "  + " if positions[16] < -5 else "    "
        g9 = "  + " if positions[15] > 5 else "  + " if positions[15] < -5 else "    "
        ga = "  + " if positions[14] > 5 else "  + " if positions[14] < -5 else "    "
        gb = "  + " if positions[13] > 5 else "  + " if positions[13] < -5 else "    "
        gc = "  + " if positions[12] > 5 else "  + " if positions[12] < -5 else "    "
        h1 = "  O " if positions[23] > 4 else "  X " if positions[23] < -4 else "    "
        h2 = "  O " if positions[22] > 4 else "  X " if positions[22] < -4 else "    "
        h3 = "  O " if positions[21] > 4 else "  X " if positions[21] < -4 else "    "
        h4 = "  O " if positions[20] > 4 else "  X " if positions[20] < -4 else "    "
        h5 = "  O " if positions[19] > 4 else "  X " if positions[19] < -4 else "    "
        h6 = "  O " if positions[18] > 4 else "  X " if positions[18] < -4 else "    "
        h7 = "  O " if positions[17] > 4 else "  X " if positions[17] < -4 else "    "
        h8 = "  O " if positions[16] > 4 else "  X " if positions[16] < -4 else "    "
        h9 = "  O " if positions[15] > 4 else "  X " if positions[15] < -4 else "    "
        ha = "  O " if positions[14] > 4 else "  X " if positions[14] < -4 else "    "
        hb = "  O " if positions[13] > 4 else "  X " if positions[13] < -4 else "    "
        hc = "  O " if positions[12] > 4 else "  X " if positions[12] < -4 else "    "
        i1 = "  O " if positions[23] > 3 else "  X " if positions[23] < -3 else "    "
        i2 = "  O " if positions[22] > 3 else "  X " if positions[22] < -3 else "    "
        i3 = "  O " if positions[21] > 3 else "  X " if positions[21] < -3 else "    "
        i4 = "  O " if positions[20] > 3 else "  X " if positions[20] < -3 else "    "
        i5 = "  O " if positions[19] > 3 else "  X " if positions[19] < -3 else "    "
        i6 = "  O " if positions[18] > 3 else "  X " if positions[18] < -3 else "    "
        i7 = "  O " if positions[17] > 3 else "  X " if positions[17] < -3 else "    "
        i8 = "  O " if positions[16] > 3 else "  X " if positions[16] < -3 else "    "
        i9 = "  O " if positions[15] > 3 else "  X " if positions[15] < -3 else "    "
        ia = "  O " if positions[14] > 3 else "  X " if positions[14] < -3 else "    "
        ib = "  O " if positions[13] > 3 else "  X " if positions[13] < -3 else "    "
        ic = "  O " if positions[12] > 3 else "  X " if positions[12] < -3 else "    "
        j1 = "  O " if positions[23] > 2 else "  X " if positions[23] < -2 else "    "
        j2 = "  O " if positions[22] > 2 else "  X " if positions[22] < -2 else "    "
        j3 = "  O " if positions[21] > 2 else "  X " if positions[21] < -2 else "    "
        j4 = "  O " if positions[20] > 2 else "  X " if positions[20] < -2 else "    "
        j5 = "  O " if positions[19] > 2 else "  X " if positions[19] < -2 else "    "
        j6 = "  O " if positions[18] > 2 else "  X " if positions[18] < -2 else "    "
        j7 = "  O " if positions[17] > 2 else "  X " if positions[17] < -2 else "    "
        j8 = "  O " if positions[16] > 2 else "  X " if positions[16] < -2 else "    "
        j9 = "  O " if positions[15] > 2 else "  X " if positions[15] < -2 else "    "
        ja = "  O " if positions[14] > 2 else "  X " if positions[14] < -2 else "    "
        jb = "  O " if positions[13] > 2 else "  X " if positions[13] < -2 else "    "
        jc = "  O " if positions[12] > 2 else "  X " if positions[12] < -2 else "    "
        k1 = "  O " if positions[23] > 1 else "  X " if positions[23] < -1 else "    "
        k2 = "  O " if positions[22] > 1 else "  X " if positions[22] < -1 else "    "
        k3 = "  O " if positions[21] > 1 else "  X " if positions[21] < -1 else "    "
        k4 = "  O " if positions[20] > 1 else "  X " if positions[20] < -1 else "    "
        k5 = "  O " if positions[19] > 1 else "  X " if positions[19] < -1 else "    "
        k6 = "  O " if positions[18] > 1 else "  X " if positions[18] < -1 else "    "
        k7 = "  O " if positions[17] > 1 else "  X " if positions[17] < -1 else "    "
        k8 = "  O " if positions[16] > 1 else "  X " if positions[16] < -1 else "    "
        k9 = "  O " if positions[15] > 1 else "  X " if positions[15] < -1 else "    "
        ka = "  O " if positions[14] > 1 else "  X " if positions[14] < -1 else "    "
        kb = "  O " if positions[13] > 1 else "  X " if positions[13] < -1 else "    "
        kc = "  O " if positions[12] > 1 else "  X " if positions[12] < -1 else "    "
        l1 = "  O " if positions[23] > 0 else "  X " if positions[23] < 0 else "    "
        l2 = "  O " if positions[22] > 0 else "  X " if positions[22] < 0 else "    "
        l3 = "  O " if positions[21] > 0 else "  X " if positions[21] < 0 else "    "
        l4 = "  O " if positions[20] > 0 else "  X " if positions[20] < 0 else "    "
        l5 = "  O " if positions[19] > 0 else "  X " if positions[19] < 0 else "    "
        l6 = "  O " if positions[18] > 0 else "  X " if positions[18] < 0 else "    "
        l7 = "  O " if positions[17] > 0 else "  X " if positions[17] < 0 else "    "
        l8 = "  O " if positions[16] > 0 else "  X " if positions[16] < 0 else "    "
        l9 = "  O " if positions[15] > 0 else "  X " if positions[15] < 0 else "    "
        la = "  O " if positions[14] > 0 else "  X " if positions[14] < 0 else "    "
        lb = "  O " if positions[13] > 0 else "  X " if positions[13] < 0 else "    "
        lc = "  O " if positions[12] > 0 else "  X " if positions[12] < 0 else "    "

        z1 = f"{positions[24] : 03d} "
        z2 = f"{positions[25] : 03d} "
        z3 = f"{positions[26] : 03d} "
        z4 = f"{positions[27] : 03d} "


        print(f"|    | 00   01   02   03   04   05 |    | 06   07   08   09   10   11 |    |")
        print(f"|----|----+----+----+----+----+----|----|----+----+----+----+----+----|----|")
        print(f"|{z4}|{a1} {a2} {a3} {a4} {a5} {a6}|    |{a7} {a8} {a9} {aa} {ab} {ac}|    |")
        print(f"|    |{b1} {b2} {b3} {b4} {b5} {b6}| P2 |{b7} {b8} {b9} {ba} {bb} {bc}|    |")
        print(f"|    |{c1} {c2} {c3} {c4} {c5} {c6}|    |{c7} {c8} {c9} {ca} {cb} {cc}|    |")
        print(f"|    |{d1} {d2} {d3} {d4} {d5} {d6}| O  |{d7} {d8} {d9} {da} {db} {dc}|    |")
        print(f"|    |{e1} {e2} {e3} {e4} {e5} {e6}|{z2}|{e7} {e8} {e9} {ea} {eb} {ec}|    |")
        print(f"|    |{f1} {f2} {f3} {f4} {f5} {f6}|    |{f7} {f8} {f9} {fa} {fb} {fc}|    |")
        print(f"|----|                             |    |                             |----|")
        print(f"|    |{g1} {g2} {g3} {g4} {g5} {g6}|    |{g7} {g8} {g9} {ga} {gb} {gc}|    |")
        print(f"|    |{h1} {h2} {h3} {h4} {h5} {h6}|{z1}|{h7} {h8} {h9} {ha} {hb} {hc}|    |")
        print(f"|    |{i1} {i2} {i3} {i4} {i5} {i6}| X  |{i7} {i8} {i9} {ia} {ib} {ic}|    |")
        print(f"|    |{j1} {j2} {j3} {j4} {j5} {j6}|    |{j7} {j8} {j9} {ja} {jb} {jc}|    |")
        print(f"|    |{k1} {k2} {k3} {k4} {k5} {k6}| P1 |{k7} {k8} {k9} {ka} {kb} {kc}|    |")
        print(f"|{z3}|{l1} {l2} {l3} {l4} {l5} {l6}|    |{l7} {l8} {l9} {la} {lb} {lc}|    |")
        print(f"|----|----+----+----+----+----+----|----|----+----+----+----+----+----|----|")
        print(f"|    | 23   22   21   20   19   18 |    | 17   16   15   14   13   12 |    |")

    def is_game_over(self) -> bool:
        """Checks if the game is over."""
        return self.positions[26] == 15 or self.positions[27] == 15

    def get_winner(self) -> int:
        """Returns the winner of the game, or 0 if there is no winner yet."""
        if self.positions[26] == 15:
            return 1
        elif self.positions[27] == 15:
            return 2
        else:
            return 0

    def _return_gym_transform(self):
        """Returns a transformed version of the board for use in a gym environment."""
        
        X = [(0,None)] * 24
        for i in range(24):
            men = self.positions[i]
            if men > 0:
                X[i] = (abs(men),0)
            elif men < 0:
                X[i] = (abs(men),1)
        B = [self.positions[25], self.positions[24]]
        O = [self.positions[27], self.positions[26]]

        return X, B, O

    def _return_tesauro_transform(self, player: int) -> list:
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