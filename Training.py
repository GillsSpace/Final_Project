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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

EPOCHS =  100 * 5000
MODEL_NAME = 'BasicTD_004'
MODEL = f'{MODEL_NAME}.pickle'
MODEL_TYPE = Models.Model_BasicTD
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

### TRAINING LOOP ###############################
print("Starting Training Loop...\n")
st = time.time()
last_step_time = time.time()
for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train_epoch()
    if (epoch + 1) % 500 == 0:
        model.time_trained_steps.append(time.time()-last_step_time+model.time_trained_steps[-1])
        last_step_time = time.time()
        print(f"Epoch {model.epochs_trained} ({epoch + 1}/{EPOCHS}) completed. Total train time = {model.time_trained_steps[-1]}, Current train time = {last_step_time - st}.")
        print("    ",end="")
        model.run_history_update_game()
    if (epoch + 1) % 5000 == 0:
        print(f"\nSaving model to {MODEL}...\n")
        Models.Model_Loader.save_model(model, MODEL)
et = time.time()
print(f"Training Loop Finished in {et-st} seconds Average EPOCH time this loop = {(et-st)/EPOCHS}.")

plot_training_history(model, MODEL_NAME)

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