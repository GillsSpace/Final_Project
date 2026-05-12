import time
import random
import gnubg_nn as gnubg

import sys
from pathlib import Path

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import scr.Logic as Logic
import scr.Models as Models

def test_gnubg_conversions():
    POSITION_SET_A = {
        'A1': [-2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        'A2': [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 0, 2, 0, 0, 0]
    }

    board = Logic.Board(POSITION_SET_A['A2'])

    board_gnu_1 = board._return_gnubg_transform(1)
    board_gnu_2 = board._return_gnubg_transform(2)

    board.render_terminal()
    print()

    print('Custom Transform:')
    print(f'P1: {board_gnu_1}')
    print(f'P2: {board_gnu_2}')
    print()

    print('From Keys:')
    print(f'P1: {gnubg.board_from_position_id('7u4OAADgc/ABYA')}')
    print(f'P2: {gnubg.board_from_position_id('4HPwAWDu7g4AAA')}')
    print()


    gnu_moves_1 = gnubg.moves(board_gnu_1,1,3,True)
    gnu_moves_2 = gnubg.moves(board_gnu_2,2,3,True)

    print('GNUBG generated moves w/ custom conversion:')
    print(f'P1 Move Set (1,2): {board._gnubg_moves_conversion(gnu_moves_1,1)}')
    print(f'P2 Move Set (2,3): {board._gnubg_moves_conversion(gnu_moves_2,2)}')

    print()
    print('Legal Move Sets from Gym:')
    print(f'P1 Move Set (1,2): {board.return_legal_moves(1,(1,2))}')
    print(f'P2 Move Set (2,3): {board.return_legal_moves(2,(2,3))}')

    print()
    print('GNUBG value of boards (win, win_gammon, win_backgammon, lose_gammon, lose_backgammon):')
    print(f'P1: {gnubg.probabilities(board_gnu_1,gnubg.p_prune)}')
    print(f'P2: {gnubg.probabilities(board_gnu_2,gnubg.p_prune)}')

    print()
    print(f'Optimal GNUBG move for P2 w/ roll (2,3): {gnubg.pub_best_move(board_gnu_2,2,3)}')

def test_training_loop(model):
    model = Models.Model_Loader.load_model(model)
    model.train_epoch()

def test_moves():
    board = Logic.Board()
    board.render_terminal()

    print(f'Legal moves for player 1 with dice (2,3):', end=' ')
    print(*board.return_legal_moves(1, (2, 3)), sep=' -- ')

    print(f'Legal moves for player 2 with dice (6,6):', end=' ')
    print(*board.return_legal_moves(2, (6, 6)), sep=' -- ')

    st = time.time()
    for _ in range(100):
        board = Logic.Board()
        player = random.choice([1, 2])
        while not board.is_game_over():
            dice = Logic.rollDice()
            legal_moves = board.return_legal_moves(player, dice)
            if legal_moves:
                move = random.choice(legal_moves)
                board.execute_move(player, move)
            player = 1 if player == 2 else 2
            board.render_terminal()
    et = time.time()
    print(f"Average time per game: {(et - st) / 100:.4f} seconds")

if __name__ == "__main__":
    MODEL = 'BasicTD_001.pickle'
    test_training_loop(MODEL)
