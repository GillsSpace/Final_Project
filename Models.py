import torch
import torchinfo
import Logic
import numpy as np
import time
import pickle
import gnubg_nn as gnubg

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

        self.time_trained = 0
        self.time_trained_steps = [0]
        self.epochs_trained = 0

        self.history_loss = []
        self.history_loss_augmented = []
        self.history_td_error = []
        self.history_accuracy = []
        self.history_game_length = []

    def forward(self, rep):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def predict(self, board:Logic.Board,player,roll):
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_all(self, board:Logic.Board,player,roll):
        raise NotImplementedError("Subclasses must implement predict_all()")
    
    def transform(self, board:Logic.Board,player):
        raise NotImplementedError("Subclasses must implement transform()")
    
    def train_epoch(self):
        raise NotImplementedError("Subclasses must implement train_epoch()")
    
    def run_diagnostic(self):
        pass

    def run_history_update_game(self):
        avg_loss = 0
        avg_loss_augmented = 0
        avg_td_error = 0
        avg_accuracy = 0
        avg_game_length = 0

        num_games = 10
        for _ in range(num_games): # average over 10 games
            board = Logic.Board()
            roll = Logic.rollDice(first=True)
            player = 1 if roll[0] > roll[1] else 2

            total_loss = 0
            total_loss_augmented = 0
            total_td_error = 0
            total_accuracy = 0
            game_length = 0

            while not board.is_game_over():
                action, pre_eval, post_eval, base_obs, next_obs = self.predict(board,player,roll)
                model_win = pre_eval[0]
                model_gammon = pre_eval[1]
                model_backgammon = pre_eval[2]

                gnubg_probs = board.return_gnubg_win_probs(player)
                gnu_win = gnubg_probs[0] if player == 1 else 1 - gnubg_probs[0]
                gnu_gammon = gnubg_probs[1] if player == 1 else gnubg_probs[3]
                gnu_backgammon = gnubg_probs[2] if player == 1 else gnubg_probs[4]

                loss = (model_win - gnu_win) ** 2
                loss_augmented = (
                    (model_win - gnu_win) ** 2 +
                    (model_gammon - gnu_gammon) ** 2 +
                    (model_backgammon - gnu_backgammon) ** 2
                )

                total_loss += loss if loss <= 1 else 1
                total_loss_augmented += loss_augmented if loss_augmented <= 1 else 1

                if len(action) > 0:
                    saved_positions = list(board.positions) 
                    gnu_rep = board._return_gnubg_transform(player)
                    flat = gnubg.pub_best_move(gnu_rep, roll[0], roll[1])
                    best_gnu_move = [
                        (flat[i], flat[i+1])
                        for i in range(0, len(flat), 2)
                    ]
                    best_gnu_move = board._gnubg_moves_conversion([(None, best_gnu_move)], player)[0]
                    board.execute_move(player, best_gnu_move)
                    after_gnu_rep = board._return_tesauro_transform(3-player)

                    if next_obs == after_gnu_rep:
                        total_accuracy += 1
                    board.positions = list(saved_positions)

                self.zero_grad()
                v_s = self.forward(torch.tensor(base_obs, dtype=torch.float32))

                with torch.no_grad():
                    if board.is_game_over():
                        reward = int(board.get_winner() == 1)
                        td_error = reward - v_s.item()
                    else:
                        v_s_next = self.forward(torch.tensor(next_obs, dtype=torch.float32))
                        td_error = v_s_next.item() - v_s.item()

                    total_td_error += td_error ** 2

                board.execute_move(player,action)
                player = 1 if player == 2 else 2
                roll = Logic.rollDice()
                game_length += 1

            if game_length > 0:
                avg_loss += total_loss / game_length
                avg_loss_augmented += total_loss_augmented / game_length
                avg_td_error += total_td_error / game_length
                avg_accuracy += total_accuracy / game_length
                avg_game_length += game_length

        avg_loss /= num_games
        avg_loss_augmented /= num_games
        avg_td_error /= num_games
        avg_accuracy /= num_games
        avg_game_length /= num_games

        print(f"Average Loss: {avg_loss:.4f}, Average Augmented Loss: {avg_loss_augmented:.4f}, Average TD Error: {avg_td_error:.4f}, Average Accuracy: {avg_accuracy:.4f}, Average Game Length: {avg_game_length:.2f}")

        self.history_loss.append(avg_loss)
        self.history_loss_augmented.append(avg_loss_augmented)
        self.history_td_error.append(avg_td_error)
        self.history_accuracy.append(avg_accuracy)
        self.history_game_length.append(avg_game_length)

class Model_BasicTD(BaseModel):
    def __init__(self, h1_size=120, h2_size=80):
        super(Model_BasicTD, self).__init__()

        self.trace_decay = 0.8
        self.learning_rate = 0.001

        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(198, h1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_size, h2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h2_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, rep):
        return self.pipeline(rep)
    
    def predict(self, board:Logic.Board,player,roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = board._return_tesauro_transform(player)
        with torch.no_grad():
            pre_eval = (self.forward(torch.tensor(pre_repr, dtype=torch.float32)).item(), 0, 0)

        if len(moves) == 0:
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            return [], pre_eval, post_eval, pre_repr, post_repr
        
        saved_positions = list(board.positions) 

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            board.positions = list(saved_positions)
            return moves[0], pre_eval, post_eval, pre_repr, post_repr

        post_repr_list = [None] * len(moves)

        for i in range(len(moves)):
            move = moves[i]
            board.execute_move(player, move)
            post_repr_list[i] = board._return_tesauro_transform(next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_eval_list = self.forward(post_repr_tensor).squeeze(dim=-1).tolist()
            win_probs = list(post_eval_list)
            post_eval_list = [(item, 0, 0) for item in post_eval_list]

        if player == 1:
            idx = np.argmax(win_probs)
        else:
            idx = np.argmin(win_probs)

        return moves[idx], pre_eval, post_eval_list[idx], pre_repr, post_repr_list[idx]
        
    def predict_all(self, board:Logic.Board,player,roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = board._return_tesauro_transform(player)
        with torch.no_grad():
            pre_eval = (self.forward(torch.tensor(pre_repr, dtype=torch.float32)).item(), 0, 0)


        if len(moves) == 0:
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            return [], pre_eval, [post_eval], pre_repr, [post_repr]
        
        saved_positions = list(board.positions) 

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            board.positions = list(saved_positions)
            return [moves[0],], pre_eval, [post_eval], pre_repr, [post_repr]

        post_repr_list = [None] * len(moves)

        for i in range(len(moves)):
            move = moves[i]
            board.execute_move(player, move)
            post_repr_list[i] = board._return_tesauro_transform(next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_eval_list = self.forward(post_repr_tensor).squeeze(dim=-1).tolist()
            post_eval_list = [(item, 0, 0) for item in post_eval_list]

        return moves, pre_eval, post_eval_list, pre_repr, post_repr_list
        
    def transform(self, board:Logic.Board,player):
        return board._return_tesauro_transform(player)
        
    def train_epoch(self):
        start_time = time.time()
        board = Logic.Board()
        roll = Logic.rollDice(first=True)
        player = 1 if roll[0] > roll[1] else 2
        
        traces = {name: torch.zeros_like(param) for name, param in self.named_parameters()}

        while not board.is_game_over():
            action, pre_eval, post_eval, pre_repr, post_repr = self.predict(board,player,roll)
            board.execute_move(player,action)
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()
            self.zero_grad()
            v_s = self.forward(torch.tensor(pre_repr, dtype=torch.float32))
            v_s.backward()
            with torch.no_grad():
                if board.is_game_over():
                    reward = int(board.get_winner() == 1)
                    td_error = reward - v_s.item()
                else:
                    v_s_next = self.forward(torch.tensor(post_repr, dtype=torch.float32))
                    td_error = v_s_next.item() - v_s.item()

                for name, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        traces[name] = (self.trace_decay * traces[name]) + p.grad
                        p.data += self.learning_rate * td_error * traces[name]

        end_time = time.time()
        self.epochs_trained += 1
        self.time_trained += (end_time - start_time)


class Model_GnubgSupervised(Model_BasicTD):
    def __init__(self, h1_size=120, h2_size=80):
        super(Model_GnubgSupervised, self).__init__(h1_size, h2_size)
        # same architecture as Model_BasicTD
        # supervised regression against gnubg win probs with adam + mse
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def train_epoch(self):
        start_time = time.time()
        board = Logic.Board()
        roll = Logic.rollDice(first=True)
        player = 1 if roll[0] > roll[1] else 2

        while not board.is_game_over():
            action, pre_eval, post_eval, pre_repr, post_repr = self.predict(board, player, roll)

            mover = player
            board.execute_move(player, action)
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()

            # supervised target from gnubg, converted to p1 perspective
            if board.is_game_over():
                target = float(board.get_winner() == 1)
            else:
                gnubg_probs = board.return_gnubg_win_probs(mover)
                gnu_win_for_mover = gnubg_probs[0]
                target = gnu_win_for_mover if mover == 1 else 1 - gnu_win_for_mover

            self.optimizer.zero_grad()
            v_s = self.forward(torch.tensor(pre_repr, dtype=torch.float32))
            target_tensor = torch.tensor([target], dtype=torch.float32)
            loss = self.loss_fn(v_s, target_tensor)
            loss.backward()
            self.optimizer.step()

        end_time = time.time()
        self.epochs_trained += 1
        self.time_trained += (end_time - start_time)


class Model_Loader:
    @staticmethod
    def save_model(model:object, filename:str):
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(filename:str) -> BaseModel:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model