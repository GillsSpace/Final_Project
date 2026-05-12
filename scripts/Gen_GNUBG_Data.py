import os
import csv
import tqdm
import gnubg_nn as gnubg

import sys
from pathlib import Path

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import scr.Logic as Logic

# create data file if it doesn't exist:
DATA_FILE = "data/training_data.csv"
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        f.write("state,win_prob\n")

N = 10_000

# Play N games against gnubg and record the state and gnubg's win probability for that state on every turn.
with open(DATA_FILE, "a", newline='') as f:
    writer = csv.writer(f) 
    for game in tqdm.tqdm(range(N)):
        board = Logic.Board()
        roll = Logic.rollDice()
        player = 1 if roll[0] > roll[1] else 2

        while not board.is_game_over():
            state = board._return_tesauro_transform(player)
            gnubg_probs = board.return_gnubg_win_probs(player)
            win_prob = gnubg_probs[0] if player == 1 else 1 - gnubg_probs[0]
            writer.writerow([state, win_prob])
            gnu_rep = board._return_gnubg_transform(player)
            flat = gnubg.pub_best_move(gnu_rep, roll[0], roll[1])
            best_gnu_move = [
                (flat[i], flat[i+1])
                for i in range(0, len(flat), 2)
            ]
            best_gnu_move = board._gnubg_moves_conversion([(None, best_gnu_move)], player)[0]
            board.execute_move(player, best_gnu_move)
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()

# Gemini was used to optimize writing to csv file.