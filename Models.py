import torch
import torchinfo
import Logic
import numpy as np
import time
import pickle
import gnubg_nn as gnubg
import os

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

        self.time_trained = 0
        self.time_trained_steps = [0]
        self.epochs_trained = 0

        self.history_loss = []
        self.history_loss_augmented = []
        self.history_last_step_loss = []
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
        avg_last_step_loss = 0
        avg_td_error = 0
        avg_accuracy = 0
        avg_game_length = 0

        num_games = 10
        for _ in range(num_games):
            board = Logic.Board()
            roll = Logic.rollDice(first=True)
            player = 1 if roll[0] > roll[1] else 2

            total_loss = 0
            total_loss_augmented = 0
            total_td_error = 0
            total_accuracy = 0
            game_length = 0
            last_step_loss = 0

            while not board.is_game_over():
                action, pre_eval, post_eval, base_obs, next_obs = self.predict(board,player,roll)
                model_win = pre_eval[0]
                model_gammon = pre_eval[1]
                model_backgammon = pre_eval[2]

                gnubg_probs = board.return_gnubg_win_probs(player)
                gnu_win = gnubg_probs[0] if player == 1 else 1 - gnubg_probs[0]
                gnu_gammon = gnubg_probs[1] if player == 1 else gnubg_probs[3]
                gnu_backgammon = gnubg_probs[2] if player == 1 else gnubg_probs[4]

                loss = abs(model_win - gnu_win)
                loss_augmented = (
                    abs(model_win - gnu_win) +
                    abs(model_gammon - gnu_gammon) +
                    abs(model_backgammon - gnu_backgammon) 
                )

                total_loss += loss if loss <= 1 else 1
                total_loss_augmented += loss_augmented if loss_augmented <= 3 else 3

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
                    after_gnu_rep = self.transform(board, 3-player)

                    if next_obs == after_gnu_rep:
                        total_accuracy += 1
                    board.positions = list(saved_positions)

                self.zero_grad()
                v_s = self.forward(torch.tensor(base_obs, dtype=torch.float32))

                with torch.no_grad():
                    if board.is_game_over():
                        reward = int(board.get_winner() == 1)
                        td_error = reward - v_s[0].item() if v_s.dim() > 0 and v_s.shape[-1] > 1 else reward - v_s.item()
                    else:
                        v_s_next = self.forward(torch.tensor(next_obs, dtype=torch.float32))
                        td_error = v_s_next[0].item() - v_s[0].item() if v_s.dim() > 0 and v_s.shape[-1] > 1 else v_s_next.item() - v_s.item()

                    total_td_error += abs(td_error)

                board.execute_move(player,action)
                player = 1 if player == 2 else 2
                roll = Logic.rollDice()
                game_length += 1

            last_step_loss = abs(model_win - gnu_win)
            last_step_loss = last_step_loss if last_step_loss <= 1 else 1

            if game_length > 0:
                avg_loss += total_loss / game_length
                avg_loss_augmented += total_loss_augmented / game_length
                avg_td_error += total_td_error / game_length
                avg_accuracy += total_accuracy / game_length
                avg_game_length += game_length
                avg_last_step_loss += last_step_loss

        avg_loss /= num_games
        avg_loss_augmented /= num_games
        avg_td_error /= num_games
        avg_accuracy /= num_games
        avg_game_length /= num_games
        avg_last_step_loss /= num_games

        print(f"Average Loss: {avg_loss:.4f}, Average Augmented Loss: {avg_loss_augmented:.4f}, Average TD Error: {avg_td_error:.4f}, Average Accuracy: {avg_accuracy:.4f}, Average Game Length: {avg_game_length:.2f}, Average Last Step Loss: {avg_last_step_loss:.4f}")

        self.history_loss.append(avg_loss)
        self.history_loss_augmented.append(avg_loss_augmented)
        self.history_td_error.append(avg_td_error)
        self.history_accuracy.append(avg_accuracy)
        self.history_game_length.append(avg_game_length)
        self.history_last_step_loss.append(avg_last_step_loss)


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


class Model_TDExplore(Model_BasicTD):
    def __init__(self, h1_size=120, h2_size=80, epsilon=0.1):
        super(Model_TDExplore, self).__init__(h1_size, h2_size)
        self.epsilon = epsilon

    def predict(self, board:Logic.Board,player,roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = board._return_tesauro_transform(player)
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

        if np.random.rand() < self.epsilon:
            idx = np.random.choice(len(moves))
        else:
            if player == 1:
                idx = np.argmax(win_probs)
            else:
                idx = np.argmin(win_probs)

        return moves[idx], pre_eval, post_eval_list[idx], pre_repr, post_repr_list[idx]


class Model_GnubgSupervised(Model_BasicTD):
    def __init__(self, h1_size=120, h2_size=80):
        super(Model_GnubgSupervised, self).__init__(h1_size, h2_size)
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


class Model_HandCrafted(BaseModel):
    # gnubg-supervised training using 19 hand-crafted features
    # instead of tesauro's 198-number encoding
    # features: pip counts, pip diff, bear-off status, blots, owned points, primes, bar/off, turn flag

    N_FEATURES = 19

    def __init__(self, h1_size=64, h2_size=32):
        super(Model_HandCrafted, self).__init__()

        self.learning_rate = 0.001
        self.loss_fn = torch.nn.MSELoss()

        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(self.N_FEATURES, h1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_size, h2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h2_size, 1),
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, rep):
        return self.pipeline(rep)

    def transform(self, board:Logic.Board, player:int) -> list:
        pos = board.positions

        p1_pip = sum((24 - i) * abs(pos[i]) for i in range(24) if pos[i] < 0)
        p1_pip += 25 * pos[24]
        p2_pip = sum((i + 1) * pos[i] for i in range(24) if pos[i] > 0)
        p2_pip += 25 * pos[25]

        pip_diff = (p1_pip - p2_pip) if player == 1 else (p2_pip - p1_pip)

        p1_bo = float(all(pos[i] >= 0 for i in range(6, 24)) and pos[24] == 0)
        p2_bo = float(all(pos[i] <= 0 for i in range(0, 18)) and pos[25] == 0)

        p1_blots = sum(1 for p in pos[:24] if p == -1) / 15
        p2_blots = sum(1 for p in pos[:24] if p == 1) / 15

        p1_blots_danger = sum(1 for i in range(18, 24) if pos[i] == -1) / 15
        p2_blots_danger = sum(1 for i in range(0, 6) if pos[i] == 1) / 15

        p1_owned = sum(1 for p in pos[:24] if p <= -2) / 24
        p2_owned = sum(1 for p in pos[:24] if p >= 2) / 24

        def longest_run(pred):
            mx, cur = 0, 0
            for p in pos[:24]:
                if pred(p):
                    cur += 1
                    mx = max(mx, cur)
                else:
                    cur = 0
            return mx

        p1_prime = longest_run(lambda p: p <= -2) / 6
        p2_prime = longest_run(lambda p: p >= 2) / 6

        p1_bar = pos[24] / 15
        p2_bar = pos[25] / 15
        p1_off = pos[26] / 15
        p2_off = pos[27] / 15

        turn = [1, 0] if player == 1 else [0, 1]

        return [
            p1_pip / 167,
            p2_pip / 167,
            pip_diff / 167,
            p1_bo,
            p2_bo,
            p1_blots,
            p2_blots,
            p1_blots_danger,
            p2_blots_danger,
            p1_owned,
            p2_owned,
            p1_prime,
            p2_prime,
            p1_bar,
            p2_bar,
            p1_off,
            p2_off,
            turn[0],
            turn[1],
        ]

    def predict(self, board:Logic.Board, player:int, roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = self.transform(board, player)
        with torch.no_grad():
            pre_eval = (self.forward(torch.tensor(pre_repr, dtype=torch.float32)).item(), 0, 0)

        if len(moves) == 0:
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            return [], pre_eval, post_eval, pre_repr, post_repr

        saved_positions = list(board.positions)

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            board.positions = list(saved_positions)
            return moves[0], pre_eval, post_eval, pre_repr, post_repr

        post_repr_list = [None] * len(moves)
        for i in range(len(moves)):
            board.execute_move(player, moves[i])
            post_repr_list[i] = self.transform(board, next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_eval_list = self.forward(post_repr_tensor).squeeze(dim=-1).tolist()
            win_probs = list(post_eval_list)
            post_eval_list = [(item, 0, 0) for item in post_eval_list]

        idx = np.argmax(win_probs) if player == 1 else np.argmin(win_probs)
        return moves[idx], pre_eval, post_eval_list[idx], pre_repr, post_repr_list[idx]

    def predict_all(self, board:Logic.Board, player:int, roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = self.transform(board, player)
        with torch.no_grad():
            pre_eval = (self.forward(torch.tensor(pre_repr, dtype=torch.float32)).item(), 0, 0)

        if len(moves) == 0:
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            return [], pre_eval, [post_eval], pre_repr, [post_repr]

        saved_positions = list(board.positions)

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), 0, 0)
            board.positions = list(saved_positions)
            return [moves[0],], pre_eval, [post_eval], pre_repr, [post_repr]

        post_repr_list = [None] * len(moves)
        for i in range(len(moves)):
            board.execute_move(player, moves[i])
            post_repr_list[i] = self.transform(board, next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_eval_list = self.forward(post_repr_tensor).squeeze(dim=-1).tolist()
            post_eval_list = [(item, 0, 0) for item in post_eval_list]

        return moves, pre_eval, post_eval_list, pre_repr, post_repr_list

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


class Model_MultiOutput(BaseModel):
    # gnubg-supervised training using 19 hand-crafted features
    # outputs all 5 gnubg probabilities: win, win_gammon, win_backgammon, lose_gammon, lose_backgammon
    # move selection uses equity score: win + win_gammon + win_backgammon - lose_gammon - lose_backgammon
    # terminal rewards correctly account for gammon and backgammon outcomes

    N_FEATURES = 19
    N_OUTPUTS = 5

    def __init__(self, h1_size=64, h2_size=32):
        super(Model_MultiOutput, self).__init__()

        self.learning_rate = 0.001
        self.loss_fn = torch.nn.MSELoss()

        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(self.N_FEATURES, h1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_size, h2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h2_size, self.N_OUTPUTS),
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, rep):
        return self.pipeline(rep)

    def _equity(self, probs) -> float:
        # probs: (win, win_gammon, win_backgammon, lose_gammon, lose_backgammon)
        # all from player 1's perspective
        return probs[0] + probs[1] + probs[2] - probs[3] - probs[4]

    def transform(self, board:Logic.Board, player:int) -> list:
        # identical to Model_HandCrafted — 19 hand-crafted features
        pos = board.positions

        p1_pip = sum((24 - i) * abs(pos[i]) for i in range(24) if pos[i] < 0)
        p1_pip += 25 * pos[24]
        p2_pip = sum((i + 1) * pos[i] for i in range(24) if pos[i] > 0)
        p2_pip += 25 * pos[25]

        pip_diff = (p1_pip - p2_pip) if player == 1 else (p2_pip - p1_pip)

        p1_bo = float(all(pos[i] >= 0 for i in range(6, 24)) and pos[24] == 0)
        p2_bo = float(all(pos[i] <= 0 for i in range(0, 18)) and pos[25] == 0)

        p1_blots = sum(1 for p in pos[:24] if p == -1) / 15
        p2_blots = sum(1 for p in pos[:24] if p == 1) / 15

        p1_blots_danger = sum(1 for i in range(18, 24) if pos[i] == -1) / 15
        p2_blots_danger = sum(1 for i in range(0, 6) if pos[i] == 1) / 15

        p1_owned = sum(1 for p in pos[:24] if p <= -2) / 24
        p2_owned = sum(1 for p in pos[:24] if p >= 2) / 24

        def longest_run(pred):
            mx, cur = 0, 0
            for p in pos[:24]:
                if pred(p):
                    cur += 1
                    mx = max(mx, cur)
                else:
                    cur = 0
            return mx

        p1_prime = longest_run(lambda p: p <= -2) / 6
        p2_prime = longest_run(lambda p: p >= 2) / 6

        p1_bar = pos[24] / 15
        p2_bar = pos[25] / 15
        p1_off = pos[26] / 15
        p2_off = pos[27] / 15

        turn = [1, 0] if player == 1 else [0, 1]

        return [
            p1_pip / 167,
            p2_pip / 167,
            pip_diff / 167,
            p1_bo,
            p2_bo,
            p1_blots,
            p2_blots,
            p1_blots_danger,
            p2_blots_danger,
            p1_owned,
            p2_owned,
            p1_prime,
            p2_prime,
            p1_bar,
            p2_bar,
            p1_off,
            p2_off,
            turn[0],
            turn[1],
        ]

    def _make_pre_eval(self, output_tensor) -> tuple:
        # output_tensor: 1D tensor of shape (5,)
        # returns 5-tuple: (win, win_gammon, win_backgammon, lose_gammon, lose_backgammon)
        vals = output_tensor.tolist()
        return tuple(vals)

    def _get_terminal_target(self, board:Logic.Board) -> list:
        # compute correct terminal target including gammon/backgammon outcomes
        winner = board.get_winner()
        pos = board.positions

        # loser has borne off zero pieces = gammon
        loser_off = pos[27] if winner == 1 else pos[26]
        is_gammon = (loser_off == 0)

        # backgammon: gammon + loser still has pieces on bar or in winner's home
        if winner == 1:
            loser_on_bar = pos[25] > 0
            loser_in_winner_home = any(pos[i] > 0 for i in range(0, 6))
        else:
            loser_on_bar = pos[24] > 0
            loser_in_winner_home = any(pos[i] < 0 for i in range(18, 24))

        is_backgammon = is_gammon and (loser_on_bar or loser_in_winner_home)

        if winner == 1:
            return [
                1.0,
                1.0 if is_gammon else 0.0,
                1.0 if is_backgammon else 0.0,
                0.0,
                0.0,
            ]
        else:
            return [
                0.0,
                0.0,
                0.0,
                1.0 if is_gammon else 0.0,
                1.0 if is_backgammon else 0.0,
            ]

    def predict(self, board:Logic.Board, player:int, roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = self.transform(board, player)
        with torch.no_grad():
            pre_out = self.forward(torch.tensor(pre_repr, dtype=torch.float32))
            pre_eval = self._make_pre_eval(pre_out)

        if len(moves) == 0:
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_out = self.forward(torch.tensor(post_repr, dtype=torch.float32))
                post_eval = self._make_pre_eval(post_out)
            return [], pre_eval, post_eval, pre_repr, post_repr

        saved_positions = list(board.positions)

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_out = self.forward(torch.tensor(post_repr, dtype=torch.float32))
                post_eval = self._make_pre_eval(post_out)
            board.positions = list(saved_positions)
            return moves[0], pre_eval, post_eval, pre_repr, post_repr

        post_repr_list = [None] * len(moves)
        for i in range(len(moves)):
            board.execute_move(player, moves[i])
            post_repr_list[i] = self.transform(board, next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_out_batch = self.forward(post_repr_tensor)
            equities = (
                post_out_batch[:, 0] +
                post_out_batch[:, 1] +
                post_out_batch[:, 2] -
                post_out_batch[:, 3] -
                post_out_batch[:, 4]
            )
            post_eval_list = [self._make_pre_eval(post_out_batch[i]) for i in range(len(moves))]

        idx = int(equities.argmax().item()) if player == 1 else int(equities.argmin().item())
        return moves[idx], pre_eval, post_eval_list[idx], pre_repr, post_repr_list[idx]

    def predict_all(self, board:Logic.Board, player:int, roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = self.transform(board, player)
        with torch.no_grad():
            pre_out = self.forward(torch.tensor(pre_repr, dtype=torch.float32))
            pre_eval = self._make_pre_eval(pre_out)

        if len(moves) == 0:
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_out = self.forward(torch.tensor(post_repr, dtype=torch.float32))
                post_eval = self._make_pre_eval(post_out)
            return [], pre_eval, [post_eval], pre_repr, [post_repr]

        saved_positions = list(board.positions)

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = self.transform(board, next_player)
            with torch.no_grad():
                post_out = self.forward(torch.tensor(post_repr, dtype=torch.float32))
                post_eval = self._make_pre_eval(post_out)
            board.positions = list(saved_positions)
            return [moves[0],], pre_eval, [post_eval], pre_repr, [post_repr]

        post_repr_list = [None] * len(moves)
        for i in range(len(moves)):
            board.execute_move(player, moves[i])
            post_repr_list[i] = self.transform(board, next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_out_batch = self.forward(post_repr_tensor)
            post_eval_list = [self._make_pre_eval(post_out_batch[i]) for i in range(len(moves))]

        return moves, pre_eval, post_eval_list, pre_repr, post_repr_list

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

            if board.is_game_over():
                # correct terminal target including gammon/backgammon
                target = self._get_terminal_target(board)
            else:
                # gnubg probs from mover's perspective, flip to p1 perspective
                gnubg_probs = board.return_gnubg_win_probs(mover)
                if mover == 1:
                    target = [
                        gnubg_probs[0],  # p1 win
                        gnubg_probs[1],  # p1 win gammon
                        gnubg_probs[2],  # p1 win backgammon
                        gnubg_probs[3],  # p1 lose gammon
                        gnubg_probs[4],  # p1 lose backgammon
                    ]
                else:
                    target = [
                        1 - gnubg_probs[0],  # p1 win = 1 - p2 win
                        gnubg_probs[3],      # p1 win gammon = p2 lose gammon
                        gnubg_probs[4],      # p1 win backgammon = p2 lose backgammon
                        gnubg_probs[1],      # p1 lose gammon = p2 win gammon
                        gnubg_probs[2],      # p1 lose backgammon = p2 win backgammon
                    ]

            self.optimizer.zero_grad()
            v_s = self.forward(torch.tensor(pre_repr, dtype=torch.float32))
            target_tensor = torch.tensor(target, dtype=torch.float32)
            loss = self.loss_fn(v_s, target_tensor)
            loss.backward()
            self.optimizer.step()

        end_time = time.time()
        self.epochs_trained += 1
        self.time_trained += (end_time - start_time)


class Model_Baseline(BaseModel):
    def __init__(self, hit_weight=2.0, blot_penalty=1.0):
        super(Model_Baseline, self).__init__()
        self.hit_weight = hit_weight
        self.blot_penalty = blot_penalty

    def forward(self, rep):
        return torch.tensor([0.5])

    def transform(self, board: Logic.Board, player):
        return board._return_tesauro_transform(player)
    
    def _count_hits(self, before, after, player):
        opponent = 3 - player
        before_bar = before[24] if opponent == 1 else before[25]
        after_bar = after[24] if opponent == 1 else after[25]
        return max(0, after_bar - before_bar)
    
    def _count_exposed_blots(self, board, player):
        blots = 0
        for pos in board.positions:
            if player == 1 and pos == -1:
                blots += 1
            elif player == 2 and pos == 1:
                blots += 1
        return blots

    def predict(self, board: Logic.Board, player, roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = self.transform(board, player)
        pre_eval = (0.5, 0, 0)

        if len(moves) == 0:
            post_repr = self.transform(board, next_player)
            return [], pre_eval, (0.5, 0, 0), pre_repr, post_repr

        best_score = -float("inf")
        best_move = None
        best_post_repr = None
        saved_positions = list(board.positions)

        for move in moves:
            board.execute_move(player, move)
            hits = self._count_hits(saved_positions, board.positions, player)
            blots = self._count_exposed_blots(board, player)
            score = self.hit_weight * hits - self.blot_penalty * blots
            if score > best_score:
                best_score = score
                best_move = move
                best_post_repr = self.transform(board, next_player)
            board.positions = list(saved_positions)

        return best_move, pre_eval, (0.5, 0, 0), pre_repr, best_post_repr

    def predict_all(self, board: Logic.Board, player, roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = self.transform(board, player)
        pre_eval = (0.5, 0, 0)

        if len(moves) == 0:
            post_repr = self.transform(board, next_player)
            return [], pre_eval, [(0.5, 0, 0)], pre_repr, [post_repr]

        saved_positions = list(board.positions)
        post_reprs = []
        evals = []

        for move in moves:
            board.execute_move(player, move)
            post_reprs.append(self.transform(board, next_player))
            evals.append((0.5, 0, 0))
            board.positions = list(saved_positions)

        return moves, pre_eval, evals, pre_repr, post_reprs
    
    def train_epoch(self):
        start_time = time.time()
        pass
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