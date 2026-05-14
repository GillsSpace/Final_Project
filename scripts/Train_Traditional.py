import os
import ast
import time
import tqdm
import torch
import torchinfo
import pandas as pd

import sys
from pathlib import Path

root_path = Path.cwd().parent if "__file__" not in globals() else Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.Logic as Logic
import src.Models as Models
from src.Logic import plot_training_history

os.environ["CUDA_VISIBLE_DEVICES"] = ""

EPOCHS =  100
MODEL_NAME = 'Traditional_001'
MODEL = f'{MODEL_NAME}.pickle'
MODEL_TYPE = Models.Model_Traditional
INPUT_SIZE = (1,198)

### LOAD IN MODEL FROM FILE #####################
print()
if os.path.exists(MODEL):
    model = Models.Model_Loader.load_model(MODEL)
    print(f"Loaded pre-trained model from {MODEL}, already trained for {model.epochs_trained} epochs.")
else:
    print(f"No pre-trained model found at {MODEL}. Starting training from scratch.")
    model = MODEL_TYPE()
    print(f"Running initial history update games...")
    print("    ",end="")
    model.run_history_update_game()

### SUMMARIZE MODEL #############################
print("Model Summary:")
torchinfo.summary(model, input_size=INPUT_SIZE)
print()

### Load Data ####################################

df = pd.read_csv('data/training_data.csv')

X_list = [ast.literal_eval(x) for x in df['state']]
Y_list = df['win_prob'].values

X_tensor = torch.tensor(X_list, dtype=torch.float32)
Y_tensor = torch.tensor(Y_list, dtype=torch.float32)

### TRAINING LOOP ###############################
print("Starting Training Loop...\n")
st = time.time()
last_step_time = time.time()
for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train_epoch(X_tensor, Y_tensor)
    if (epoch + 1) % 10 == 0:
        model.time_trained_steps.append(time.time()-last_step_time+model.time_trained_steps[-1])
        last_step_time = time.time()
        print(f"Epoch {model.epochs_trained} ({epoch + 1}/{EPOCHS}) completed. Total train time = {model.time_trained_steps[-1]}, Current train time = {last_step_time - st}.")
        print("    ",end="")
        model.run_history_update_game()
    if (epoch + 1) % 100 == 0:
        print(f"\nSaving model to {MODEL}...\n")
        Models.Model_Loader.save_model(model, MODEL)
et = time.time()
print(f"Training Loop Finished in {et-st} seconds Average EPOCH time this loop = {(et-st)/EPOCHS}.")

plot_training_history(model, MODEL_NAME)

baseline_model = Models.Model_Loader.load_model('Baseline_001.pickle')
num_games = 1000
print(f"\nEvaluating model against baseline for {num_games} games...")
win_rate = 0
for _ in range(num_games): # have the model play 1000 games against the baseline and track win rate
    # P1: trained model, P2: baseline
    board = Logic.Board()
    roll = Logic.rollDice(first=True)
    player = 1 if roll[0] > roll[1] else 2
    models = {1: model, 2: baseline_model}

    while not board.is_game_over():
        action, _, post_eval, _, _ = models[player].predict(board,player,roll)
        board.execute_move(player,action)
        player = 3 - player
        roll = Logic.rollDice()
    if board.get_winner() == 1:
        win_rate += 1
print(f"Model win rate against baseline: {win_rate/num_games:.2f}")

### SAVE FINAL MODEL ############################
print(f"Saving model to {MODEL}...")
Models.Model_Loader.save_model(model, MODEL)

# print("\n predictions for opening moves:")
# board = Logic.Board()
# for roll in Logic.ROLLS:
#     if roll[0] == roll[1]:
#         continue
#     val, action, _, _ = model.predict(board,roll,1)
#     print(f"Roll: {roll} --> Move: {action} ({val})")