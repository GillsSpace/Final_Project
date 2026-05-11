import torch
import torchinfo
import scr.Logic as Logic
import numpy as np
import time
import scr.Models as Models
import os
import scripts.Validation as Validation
from scr.Logic import plot_training_history

os.environ["CUDA_VISIBLE_DEVICES"] = ""

EPOCHS = 10 * 5000
MODEL = 'MultiOutput_001.pickle'
MODEL_TYPE = Models.Model_MultiOutput
INPUT_SIZE = (1, 19)

print()
if os.path.exists(MODEL):
    model = Models.Model_Loader.load_model(MODEL)
    print(f"Loaded pre-trained model from {MODEL}, already trained for {model.epochs_trained} epochs.")
else:
    print(f"No pre-trained model found at {MODEL}. Starting training from scratch.")
    model = MODEL_TYPE()

#summarize
print("Model Summary:")
torchinfo.summary(model, input_size=INPUT_SIZE)
print()

#training loop
print("Starting Training Loop...\n")
st = time.time()
last_step_time = time.time()
for epoch in range(EPOCHS):
    model.train_epoch()
    if (epoch + 1) % 500 == 0:
        model.time_trained_steps.append(time.time()-last_step_time+model.time_trained_steps[-1])
        model.run_history_update_game()
        last_step_time = time.time()
        print(f"Epoch {model.epochs_trained} ({epoch + 1}/{EPOCHS}) completed. Total train time = {model.time_trained_steps[-1]}, Current train time = {last_step_time - st}.")
    if (epoch + 1) % 5000 == 0:
        print(f"\nSaving model to {MODEL}...\n")
        Models.Model_Loader.save_model(model, MODEL)
et = time.time()
print(f"Training Loop Finished in {et-st} seconds Average EPOCH time this loop = {(et-st)/EPOCHS}.")

plot_training_history(model)

#save
print(f"Saving model to {MODEL}...")
Models.Model_Loader.save_model(model, MODEL)