import gnubg_nn as gnubg
import numpy as np

import sys
from pathlib import Path

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.Logic as Logic
import src.Models as Models

POSITION_SET_A = {
    'A1': [-2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    'A2': [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    'A3': [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2, -2, -2, -2, -2, 2, 0, 0, 0],
}

def _format_eval(post_eval):
    # post_eval is (win, gammon_win, bg_win, gammon_loss, bg_loss)
    win = post_eval[0]
    gammon_win = post_eval[1]
    gammon_loss = post_eval[3]
    return f"win: {win:.1%}  |  gammon win: {gammon_win:.1%}  |  gammon loss: {gammon_loss:.1%}"

def run_exhibition_game_terminal(model):
    if type(model) == str:
        model = Models.Model_Loader.load_model(model)
    board = Logic.Board()
    roll = Logic.rollDice(True)
    player = 1 if roll[0] > roll[1] else 2

    print(f"Starting Game...")
    board.render_terminal()
    print()
    print(f"Player {player} won opening roll with {roll}")
    while not board.is_game_over():
        action, _, post_eval, _, _ = model.predict(board,player,roll)
        print(f"Player {player} with roll {roll} makes move {action}  ({_format_eval(post_eval)})...")
        board.execute_move(player,action)
        board.render_terminal()
        print()
        player = 1 if player == 2 else 2
        roll = Logic.rollDice()
        
def play_in_terminal(model):
    #P1 = Human, P2 = Bot
    model = Models.Model_Loader.load_model(model)

    board = Logic.Board()
    roll = Logic.rollDice(True)
    player = 1 if roll[0] > roll[1] else 2

    print(f"Starting Game...")
    print(f"Playing against {model._get_name()} trained for {model.epochs_trained} epochs\n.")
    board.render_terminal()
    print(f"Player {player} won opening roll with {roll}\n")

    while not board.is_game_over():
        if player == 2:
            action, _, post_eval, _, _ = model.predict(board,player,roll)
            print(f"Player {player} with roll {roll} makes move {action}  ({_format_eval(post_eval)})...")
            board.execute_move(player,action)
            board.render_terminal()
            print()
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()
        else:
            possible_actions = board.return_legal_moves(1,roll)
            actions, pre_evals, post_evals, _, _ = model.predict_all(board,player,roll)
            num_possible_actions = len(possible_actions)
            print(f"Roll is {roll}, please select move from choices below (type 'm' to see all):")
            if num_possible_actions > 0:
                win_probs = [item[0] for item in post_evals]
                ranking = np.argsort(win_probs)[::-1]
                top_idx = list(ranking[:5])

                top_moves = [actions[i] for i in top_idx]
                top_evals = [post_evals[i] for i in top_idx]
                for i in range(len(top_moves)):
                    print(f'{i+1} ({top_evals[i][0]}) --> {top_moves[i]}')
                choice = input('Selected Move: ')
                if choice == 'm':
                    for i in range(len(possible_actions)):
                        print(f'{i+1} --> {possible_actions[i]}')
                    choice = int(input('Selected Move: ')) - 1
                    action = possible_actions[choice]
                    board.execute_move(player,action)
                else:
                    choice = int(choice) - 1
                    action = top_moves[choice]
                    board.execute_move(player,action)
            else:
                _ = input('No Possible Moves. Press ENTER to continue.')
            board.render_terminal()
            print()
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()

def test_opening_moves(model):
    model = Models.Model_Loader.load_model(model)

    board = Logic.Board()

    print(f"\nTesting model predictions for opening moves...\n")

    board.render_terminal()

    print('\nMove predictions and evaluation for player 1:\n')
    
    for roll in Logic.FIRST_ROLLS:
        action, _, post_eval, _, _ = model.predict(board,1,roll)
        print(f'    {roll} ----> {action}  ({_format_eval(post_eval)})')

def play_x_moves(model, x=3):
    model = Models.Model_Loader.load_model(model)

    board = Logic.Board()
    roll = Logic.rollDice(True)
    player = 1 if roll[0] > roll[1] else 2

    print(f"\nStarting Game of {x} Moves...")
    board.render_terminal()
    print(f"Player {player} won opening roll with {roll}\n")
    move_count = 0
    while not board.is_game_over() and move_count < x:
        action, _, post_eval, _, _ = model.predict(board,player,roll)
        print(f"Player {player} with roll {roll} makes move {action}  ({_format_eval(post_eval)})...")
        board.execute_move(player,action)
        board.render_terminal()
        print()
        player = 1 if player == 2 else 2
        roll = Logic.rollDice()
        move_count += 1


if __name__ == "__main__":
    sys.modules['Models'] = Models
    MODEL = 'models/BasicTD_Final.pickle'
    play_in_terminal(MODEL)