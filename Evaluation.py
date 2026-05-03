import torch
import torchinfo
import Logic
import numpy as np
import time
import Models
import os
import Validation
from Logic import plot_training_history
import tqdm
import gnubg_nn as gnubg
import matplotlib.pyplot as plt
import random
from Training_all import train_all

model_types = ["BasicTD", "GnubgSupervised", "HandCrafted", "MultiOutput"]
model_names = [f"{model_type}_final" for model_type in model_types]
metrics = ["time_trained", "history_loss", "history_loss_augmented", "history_accuracy"]
train_all(model_types, model_names, max_epochs=10*5000)

baseline_model = Models.Model_Loader.load_model('Baseline_001.pickle')
num_games = 5000
win_rates_baseline = []
win_rates_gnubg = []
win_rates_models = np.zeros((len(model_types), len(model_types)))
metrics_history = {model_name: {metric: [] for metric in metrics} for model_name in model_names}

for i, model_name in enumerate(model_names):
    MODEL = f'{model_name}.pickle'
    model = Models.Model_Loader.load_model(MODEL)
    for metric in metrics:
        metrics_history[model_name][metric] = getattr(model, metric)

#     print(f"\nEvaluating {model_name} against baseline for {num_games} games...")
#     win_rate_baseline = 0
#     for _ in range(num_games): # have the model play games against the baseline and track win rate
#         board = Logic.Board()
#         roll = Logic.rollDice(first=True)
#         player_model = random.choice([1, 2])
#         player = 1 if roll[0] > roll[1] else 2
#         models = {1: model if player_model == 1 else baseline_model,
#                   2: model if player_model == 2 else baseline_model}

#         while not board.is_game_over():
#             action, _, post_eval, _, _ = models[player].predict(board,player,roll)
#             board.execute_move(player,action)
#             player = 3 - player
#             roll = Logic.rollDice()
#         if board.get_winner() == player_model:
#             win_rate_baseline += 1
#     print(f"Model win rate against baseline: {win_rate_baseline/num_games:.2f}")
#     win_rates_baseline.append(win_rate_baseline/num_games)

#     print(f"\nEvaluating {model_name} against GNUBG for {num_games} games...")
#     win_rate_gnubg = 0
#     for _ in range(num_games): # have the model play games against the GNUBG and track win rate
#         # P1: trained model, P2: GNUBG
#         board = Logic.Board()
#         roll = Logic.rollDice(first=True)
#         player_model = random.choice([1, 2])
#         player = 1 if roll[0] > roll[1] else 2

#         while not board.is_game_over():
#             if player == player_model:
#                 action, _, post_eval, _, _ = model.predict(board,player,roll)
#                 board.execute_move(player,action)
#             else:
#                 gnu_rep = board._return_gnubg_transform(player)
#                 flat = gnubg.pub_best_move(gnu_rep, roll[0], roll[1])
#                 best_gnu_move = [
#                     (flat[i], flat[i+1])
#                     for i in range(0, len(flat), 2)
#                 ]
#                 best_gnu_move = board._gnubg_moves_conversion([(None, best_gnu_move)], player)[0]
#                 board.execute_move(player, best_gnu_move)
#             player = 3 - player
#             roll = Logic.rollDice()
#         if board.get_winner() == player_model:
#             win_rate_gnubg += 1
#     print(f"Model win rate against GNUBG: {win_rate_gnubg/num_games:.2f}")
#     win_rates_gnubg.append(win_rate_gnubg/num_games)

#     for j, model_name2 in enumerate(model_names[:i+1]):
#         MODEL2 = f'{model_name2}.pickle'
#         model2 = Models.Model_Loader.load_model(MODEL2)
#         print(f"\nEvaluating {model_name} against {model_name2} for {num_games} games...")
#         win_rate_model2 = 0
#         for _ in range(num_games): # have the model play games against the other model and track win rate
#             board = Logic.Board()
#             roll = Logic.rollDice(first=True)
#             player_model1 = random.choice([1, 2])
#             player = 1 if roll[0] > roll[1] else 2
#             models = {1: model if player_model1 == 1 else model2,
#                       2: model if player_model1 == 2 else model2}

#             while not board.is_game_over():
#                 action, _, post_eval, _, _ = models[player].predict(board,player,roll)
#                 board.execute_move(player,action)
#                 player = 3 - player
#                 roll = Logic.rollDice()
#             if board.get_winner() == player_model1:
#                 win_rate_model2 += 1
#         win_rate_model2 = win_rate_model2 / num_games
#         win_rates_models[i][j] = win_rate_model2
#         print(f"Model win rate against {model_name2}: {win_rate_model2:.2f}")

# for i in range(len(model_types)):
#     for j in range(len(model_types)):
#         if i > j:
#             win_rates_models[j][i] = 1 - win_rates_models[i][j]

def plot_win_rates():
    x = np.arange(len(model_types)) 
    width = 0.35 

    fig, ax = plt.subplots()

    bars1 = ax.bar(x - width/2, win_rates_baseline, width, label='vs Baseline')
    bars2 = ax.bar(x + width/2, win_rates_gnubg, width, label='vs GNUBG')

    ax.set_ylabel('Win Rate')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_types, rotation=30)
    ax.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("plots/win_rates.png", dpi=300)
    plt.show()

def sma(values, window=10):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i+1]
        smoothed.append(sum(window_vals) / len(window_vals))
    return smoothed

def plot_metric(metric, alpha=0.1):
    plt.figure()
    
    for model_name in model_names:
        values = metrics_history[model_name][metric]
        smoothed = sma(values, window=10)
        plt.plot(smoothed, label=model_name)

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*500):,}'))
    plt.xlabel("Training Steps")
    plt.ylabel(metric)
    plt.title(f"{metric} over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{metric}.png", dpi=300)
    plt.show()

def plot_win_rate_matrix():
    fig, ax = plt.subplots()

    im = ax.imshow(win_rates_models, cmap="coolwarm", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(model_types)))
    ax.set_yticks(np.arange(len(model_types)))
    ax.set_xticklabels(model_types, rotation=45)
    ax.set_yticklabels(model_types)

    # Annotate values
    for i in range(len(model_types)):
        for j in range(len(model_types)):
            ax.text(j, i, f"{win_rates_models[i, j]:.2f}",
                    ha="center", va="center")

    plt.title("Model vs Model Win Rates")
    plt.xlabel("Opponent")
    plt.ylabel("Model")
    plt.colorbar(im)

    plt.tight_layout()
    plt.savefig("plots/winrate_matrix.png", dpi=300)
    plt.show()

# plot_win_rates()
for metric in ["history_loss", "history_loss_augmented", "history_accuracy"]:
    plot_metric(metric)
# plot_win_rate_matrix()

# plt.figure()

# model_times = [metrics_history[m]["time_trained"] for m in model_names]

# plt.bar(model_names, model_times)

# plt.xlabel("Model")
# plt.ylabel("Time Trained")
# plt.title("Training Time per Model")

# plt.tight_layout()
# plt.savefig("plots/time_trained.png", dpi=300)
# plt.show()