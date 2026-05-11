import torch
import torchinfo
import scr.Logic as Logic
import numpy as np
import time
import scr.Models as Models
import os
import scripts.Validation as Validation
from scr.Logic import plot_training_history
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def train_all(model_types, model_names, max_epochs=100*5000):
    for model_type, model_name in zip(model_types, model_names):
        if model_type == "Baseline":
            print(f"\n{'='*20} Skipping training for Baseline model {'='*20}\n")
            Models.Model_Loader.save_model(Models.Model_Baseline(), f'{model_name}.pickle')
            continue

        print(f"\n{'='*20} Training {model_type} up to {max_epochs} total epochs {'='*20}\n")
        MODEL = f'{model_name}.pickle'
        MODEL_TYPE = getattr(Models, f'Model_{model_type}')
        INPUT_SIZE = (1,198) if model_type in ["BasicTD", "GnubgSupervised", "EpsilonGreedy"] else (1,19)

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

        if model.epochs_trained > max_epochs:
            raise ValueError(
                f"Model {model_name} already trained for {model.epochs_trained} epochs, "
                f"which exceeds max_epochs={max_epochs}."
            )

        remaining_epochs = max_epochs - model.epochs_trained

        if remaining_epochs == 0:
            print(f"Model {model_name} already fully trained ({max_epochs} epochs). Skipping.")
            continue

        print(f"Training for {remaining_epochs} more epochs...\n")

        ### SUMMARIZE MODEL #############################
        print("Model Summary:")
        torchinfo.summary(model, input_size=INPUT_SIZE)
        print()

        ### TRAINING LOOP ###############################
        print("Starting Training Loop...\n")
        st = time.time()
        last_step_time = time.time()
        for epoch in tqdm.tqdm(range(remaining_epochs)):
            model.train_epoch()
            if (epoch + 1) % 500 == 0:
                model.time_trained_steps.append(time.time()-last_step_time+model.time_trained_steps[-1])
                last_step_time = time.time()
                print(f"Epoch {model.epochs_trained} ({epoch + 1}/{remaining_epochs}) completed. Total train time = {model.time_trained_steps[-1]}, Current train time = {last_step_time - st}.")
                print("    ",end="")
                model.run_history_update_game()
            if (epoch + 1) % 5000 == 0:
                print(f"\nSaving model to {MODEL}...\n")
                Models.Model_Loader.save_model(model, MODEL)
        et = time.time()
        print(f"Training Loop Finished in {et-st} seconds Average EPOCH time this loop = {(et-st)/remaining_epochs}.")

        plot_training_history(model, model_name)

        print(f"Saving model to {MODEL}...")
        Models.Model_Loader.save_model(model, MODEL)