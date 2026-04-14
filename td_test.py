import torch
import torchinfo
import Logic
import numpy as np
import time
import Models
import os

EPOCHS = 100000
MODEL = 'v1_001.pickle'

print("Model Summary:")

print(torchinfo.summary(Models.Model_v1(), input_size=(1, 198)))

if os.path.exists(MODEL):
    model = Models.Model_Loader.load_model(MODEL)
    print(f"Loaded pre-trained model from {MODEL}, already trained for {model.epochs_trained} epochs.")
else:
    print(f"No pre-trained model found at {MODEL}. Starting training from scratch.")
    model = Models.Model_v1()

print("\n predictions for opening moves:")
board = Logic.Board()
for roll in Logic.ROLLS:
    if roll[0] == roll[1]:
        continue
    val, action, _, _ = model.predict(board,roll,1)
    print(f"Roll: {roll} --> Move: {action} ({val})")

print()
st = time.time()
for epoch in range(EPOCHS):
    model.run_training_game()
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} completed in {time.time() - st:.2f} seconds.")
et = time.time()

print(f"Training Finished in {et-st} seconds Average EPOCH = {(et-st)/EPOCHS}")
print("predictions for opening moves:")
board = Logic.Board()
board.render_terminal()
for roll in Logic.ROLLS:
    if roll[0] == roll[1]:
        continue
    val, action, _, _ = model.predict(board,roll,1)
    print(f"Roll: {roll} --> Move: {action} ({val})")

print(f"Saving model to {MODEL}...")
Models.Model_Loader.save_model(model, MODEL)