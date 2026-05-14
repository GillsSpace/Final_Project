import pickle
import numpy as np
import random
import gnubg_nn as gnubg
import matplotlib.pyplot as plt
import sys
from pathlib import Path

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.Logic as Logic
import src.Models as Models

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

        self.metrics = ["history_loss", "history_loss_augmented", "history_accuracy", "history_last_step_loss", "history_td_error", "history_game_length"]

        self.metrics_history = {
            name: {metric: getattr(self.models[name], metric, [])
                for metric in self.metrics}
            for name in model_names
        }

    def play_game(self, model1, model2, max_moves=500):
        board = Logic.Board()
        roll = Logic.rollDice(first=True)

        player_model1 = random.choice([1, 2])
        player = 1 if roll[0] > roll[1] else 2

        models = {
            1: model1 if player_model1 == 1 else model2,
            2: model1 if player_model1 == 2 else model2
        }
        moves = 0
        while not board.is_game_over():
            if moves >= max_moves:
                return None  # Draw / timeout
            action, _, _, _, _ = models[player].predict(board, player, roll)
            board.execute_move(player, action)
            player = 3 - player
            roll = Logic.rollDice()
            moves += 1

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
                timeouts = 0
                for _ in range(self.num_games):
                    result = self.play_game(model1, model2)
                    if result is None:
                        timeouts += 1
                    else:
                        wins += result

                # Timed-out games count as 0.5 each
                rate = (wins + timeouts * 0.5) / self.num_games
                self.win_matrix[i][j] = rate
                self.win_matrix[j][i] = 1 - rate

                print(f"{name1} vs {name2}: {rate:.2f} ({timeouts} timeouts)")

    def run_all(self):
        self.evaluate_vs_baseline()
        self.evaluate_vs_gnubg()
        self.evaluate_models()

    def plot_win_rates(self, save_path="plots/win_rates.png"):
        x = np.arange(len(self.model_names))
        width = 0.35

        baseline = [self.win_vs_baseline[m] for m in self.model_names]
        gnubg = [self.win_vs_gnubg[m] for m in self.model_names]

        display_names = [name.replace("models/", "").replace("_Final", "") for name in self.model_names]

        fig, ax = plt.subplots()

        bars1 = ax.bar(x - width/2, baseline, width, label='vs Baseline')
        bars2 = ax.bar(x + width/2, gnubg, width, label='vs GNUBG')
        max_height = max(max(baseline), max(gnubg))
        ax.set_ylim(0, max_height * 1.08)  

        ax.set_ylabel('Win Rate')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30)
        ax.legend(
            loc='center',
            bbox_to_anchor=(0.8, 0.5)
        )

        # for bar in bars1 + bars2:
        #     height = bar.get_height()
        #     ax.annotate(f'{height:.2f}',
        #                 xy=(bar.get_x() + bar.get_width() / 2, height),
        #                 xytext=(0, 3),
        #                 textcoords="offset points",
        #                 ha='center', va='bottom')
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(-5, 3),
                textcoords="offset points",
                ha='center',
                va='bottom'
            )

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(5, 10),  
                textcoords="offset points",
                ha='center',
                va='bottom'
            )

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
            if not values or "Traditional" in model_name: # Traditional model doesn't have training history
                continue
            display_name = model_name.replace("models/", "").replace("_Final", "")

            smoothed = self.sma(values, window=10)
            plt.plot(smoothed, label=display_name)

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

        im = ax.imshow(self.win_matrix, cmap="coolwarm", vmin=0, vmax=1, alpha=0.8)

        display_names = [name.replace("models/", "").replace("_Final", "") for name in self.model_names]

        ax.set_xticks(np.arange(len(self.model_names)))
        ax.set_yticks(np.arange(len(self.model_names)))
        ax.set_xticklabels(display_names, rotation=45)
        ax.set_yticklabels(display_names)

        for i in range(len(self.model_names)):
            for j in range(len(self.model_names)):
                val = self.win_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

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