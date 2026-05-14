import pickle
import os
import numpy as np
import random
import sys
from pathlib import Path
import gnubg_nn as gnubg
import matplotlib.pyplot as plt
import types

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from Training_all import train_all
import src.Logic as Logic
import src.Models as Models
from src.Tournament import Tournament

model_types = ["BasicTD", "BigTD", "SmallTD", "TDExplore", "GnubgSupervised", "HandCrafted", "BoardStandard", "MultiOutput", "Traditional"]
model_names = [f"models/{model_type}_Final" for model_type in model_types]
sys.modules['Models'] = Models
scr_pkg = types.ModuleType("scr")
scr_pkg.Models = Models
scr_pkg.Logic = Logic
sys.modules["scr"] = scr_pkg
sys.modules["scr.Models"] = Models
train_all(model_types, model_names, max_epochs=20*5000)
    
TOURNAMENT_PATH = "tournament_final_10000.pkl"

if os.path.exists(TOURNAMENT_PATH):
    print("Loading existing tournament...")
    tournament = Tournament.load(TOURNAMENT_PATH)
else:
    print("Running new tournament...")
    baseline_model = Models.Model_Loader.load_model('models/Baseline_Final.pickle')

    tournament = Tournament(
        model_names=model_names,
        baseline_model=baseline_model,
        num_games=10000
    )

    tournament.run_all()
    tournament.save(TOURNAMENT_PATH)

tournament.plot_win_rates()

for metric, label in [
    ("history_loss", "Loss"),
    ("history_loss_augmented", "Augmented Loss"),
    ("history_accuracy", "Accuracy"),
    ("history_last_step_loss", "Last Step Loss"),
    ("history_td_error", "TD Error"),
    ("history_game_length", "Game Length")
]:
    tournament.plot_metric(metric, label)

tournament.plot_win_rate_matrix()