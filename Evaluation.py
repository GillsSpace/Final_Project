import pickle
import os
import numpy as np
import random
import Logic
import Models
import gnubg_nn as gnubg
import matplotlib.pyplot as plt
from Training_all import train_all

model_types = ["BasicTD", "GnubgSupervised", "HandCrafted", "MultiOutput"]
model_names = [f"{model_type}_final" for model_type in model_types]
train_all(model_types, model_names, max_epochs=10*5000)
num_games = 5000

class Tournament:
    def __init__(self, model_names, baseline_model, num_games=5000):
        self.model_names = model_names
        self.num_games = num_games
        self.baseline_model = baseline_model

        self.models = {
            name: Models.Model_Loader.load_model(f"{name}.pickle")
            for name in model_names
        }

        self.win_vs_baseline = {name: None for name in model_names}
        self.win_vs_gnubg = {name: None for name in model_names}
        self.win_matrix = np.full((len(model_names), len(model_names)), np.nan)

        self.metrics = ["history_loss", "history_loss_augmented", "history_accuracy"]

        self.metrics_history = {
            name: {metric: getattr(self.models[name], metric, [])
                for metric in self.metrics}
            for name in model_names
        }

    def play_game(self, model1, model2):
        board = Logic.Board()
        roll = Logic.rollDice(first=True)

        player_model1 = random.choice([1, 2])
        player = 1 if roll[0] > roll[1] else 2

        models = {
            1: model1 if player_model1 == 1 else model2,
            2: model1 if player_model1 == 2 else model2
        }

        while not board.is_game_over():
            action, _, _, _, _ = models[player].predict(board, player, roll)
            board.execute_move(player, action)
            player = 3 - player
            roll = Logic.rollDice()

        return board.get_winner() == player_model1

    def evaluate_vs_baseline(self):
        for name, model in self.models.items():
            wins = 0
            for _ in range(self.num_games):
                if self.play_game(model, self.baseline_model):
                    wins += 1
            self.win_vs_baseline[name] = wins / self.num_games
            print(f"{name} vs baseline: {self.win_vs_baseline[name]:.2f}")

    def evaluate_vs_gnubg(self):
        for name, model in self.models.items():
            wins = 0
            for _ in range(self.num_games):
                if self.play_vs_gnubg(model):
                    wins += 1
            self.win_vs_gnubg[name] = wins / self.num_games
            print(f"{name} vs GNUBG: {self.win_vs_gnubg[name]:.2f}")

    def play_vs_gnubg(self, model):
        board = Logic.Board()
        roll = Logic.rollDice(first=True)

        player_model = random.choice([1, 2])
        player = 1 if roll[0] > roll[1] else 2

        while not board.is_game_over():
            if player == player_model:
                action, _, _, _, _ = model.predict(board, player, roll)
                board.execute_move(player, action)
            else:
                gnu_rep = board._return_gnubg_transform(player)
                flat = gnubg.pub_best_move(gnu_rep, roll[0], roll[1])

                best_move = [
                    (flat[i], flat[i+1])
                    for i in range(0, len(flat), 2)
                ]
                best_move = board._gnubg_moves_conversion([(None, best_move)], player)[0]

                board.execute_move(player, best_move)

            player = 3 - player
            roll = Logic.rollDice()

        return board.get_winner() == player_model

    def evaluate_models(self):
        n = len(self.model_names)

        for i, name1 in enumerate(self.model_names):
            for j, name2 in enumerate(self.model_names):
                if i < j:
                    continue 

                model1 = self.models[name1]
                model2 = self.models[name2]

                wins = 0
                for _ in range(self.num_games):
                    if self.play_game(model1, model2):
                        wins += 1

                rate = wins / self.num_games
                self.win_matrix[i][j] = rate
                self.win_matrix[j][i] = 1 - rate

                print(f"{name1} vs {name2}: {rate:.2f}")

    def run_all(self):
        self.evaluate_vs_baseline()
        self.evaluate_vs_gnubg()
        self.evaluate_models()

    def plot_win_rates(self, save_path="plots/win_rates.png"):
        x = np.arange(len(self.model_names))
        width = 0.35

        baseline = [self.win_vs_baseline[m] for m in self.model_names]
        gnubg = [self.win_vs_gnubg[m] for m in self.model_names]

        fig, ax = plt.subplots()

        bars1 = ax.bar(x - width/2, baseline, width, label='vs Baseline')
        bars2 = ax.bar(x + width/2, gnubg, width, label='vs GNUBG')

        ax.set_ylabel('Win Rate')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names, rotation=30)
        ax.legend()

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    def sma(self, values, window=10):
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(sum(values[start:i+1]) / (i - start + 1))
        return smoothed


    def plot_metric(self, metric, label):
        plt.figure()

        for model_name in self.model_names:
            values = self.metrics_history[model_name].get(metric, [])
            if not values:
                continue

            smoothed = self.sma(values, window=10)
            plt.plot(smoothed, label=model_name)

        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{int(x*500):,}')
        )

        plt.xlabel("Training Steps")
        plt.ylabel(label)
        plt.title(f"{label} over Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{metric}.png", dpi=300)
        plt.show()

    def plot_win_rate_matrix(self, save_path="plots/winrate_matrix.png"):
        fig, ax = plt.subplots()

        im = ax.imshow(self.win_matrix, cmap="coolwarm", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(self.model_names)))
        ax.set_yticks(np.arange(len(self.model_names)))
        ax.set_xticklabels(self.model_names, rotation=45)
        ax.set_yticklabels(self.model_names)

        for i in range(len(self.model_names)):
            for j in range(len(self.model_names)):
                val = self.win_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center")

        plt.title("Model vs Model Win Rates")
        plt.xlabel("Opponent")
        plt.ylabel("Model")
        plt.colorbar(im)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
TOURNAMENT_PATH = "tournament_001_10000.pkl"

if os.path.exists(TOURNAMENT_PATH):
    print("Loading existing tournament...")
    tournament = Tournament.load(TOURNAMENT_PATH)
else:
    print("Running new tournament...")
    baseline_model = Models.Model_Loader.load_model('Baseline_001.pickle')

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
    ("history_accuracy", "Accuracy")
]:
    tournament.plot_metric(metric, label)

tournament.plot_win_rate_matrix()