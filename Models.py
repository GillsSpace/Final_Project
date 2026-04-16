import torch
import torchinfo
import Logic
import numpy as np
import time
import pickle

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
        pass


class Model_BasicTD(BaseModel):
    def __init__(self):
        super(Model_BasicTD, self).__init__()

        self.trace_decay = 0.8
        self.learning_rate = 0.001

        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(198, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, rep):
        return self.pipeline(rep)
    
    def predict(self, board:Logic.Board,player,roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = board._return_tesauro_transform(player)
        pre_eval = (self.forward(torch.tensor(pre_repr, dtype=torch.float32)).item(), ) * 3

        if len(moves) == 0:
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), ) * 3
            return [], pre_eval, post_eval, pre_repr, post_repr
        
        saved_positions = list(board.positions) 

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), ) * 3
            board.positions = list(saved_positions)
            return list(moves[0]), pre_eval, post_eval, pre_repr, post_repr

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
            post_eval_list = [(item, )*3 for item in post_eval_list]

        if player == 1:
            idx = np.argmax(win_probs)
        else:
            idx = np.argmin(win_probs)

        return moves[idx], pre_eval, post_eval_list[idx], pre_repr, post_repr_list[idx]
        
    def predict_all(self, board:Logic.Board,player,roll):
        moves = board.return_legal_moves(player, roll)
        next_player = 3 - player
        pre_repr = board._return_tesauro_transform(player)
        pre_eval = (self.forward(torch.tensor(pre_repr, dtype=torch.float32)).item(), ) * 3

        if len(moves) == 0:
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), ) * 3
            return [], pre_eval, [post_eval], pre_repr, [post_repr]
        
        saved_positions = list(board.positions) 

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            post_repr = board._return_tesauro_transform(next_player)
            with torch.no_grad():
                post_eval = (self.forward(torch.tensor(post_repr, dtype=torch.float32)).item(), ) * 3
            board.positions = list(saved_positions)
            return list(moves[0]), pre_eval, [post_eval], pre_repr, [post_repr]

        post_repr_list = [None] * len(moves)

        for i in range(len(moves)):
            move = moves[i]
            board.execute_move(player, move)
            post_repr_list[i] = board._return_tesauro_transform(next_player)
            board.positions = list(saved_positions)

        with torch.no_grad():
            post_repr_tensor = torch.tensor(post_repr_list, dtype=torch.float32)
            post_eval_list = self.forward(post_repr_tensor).squeeze(dim=-1).tolist()
            post_eval_list = [(item,)*3 for item in post_eval_list]

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


class Model_v1(torch.nn.Module):
    def __init__(self, trace_decay=0.8, learning_rate=0.001):
        super(Model_v1, self).__init__()

        self.trace_decay = trace_decay
        self.learning_rate = learning_rate

        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(198, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid()
        )

        self.epochs_trained = 0

    def forward(self, x):
        return self.pipeline(x)
    
    def predict(self, board:Logic.Board, dice:tuple[int, int], player:int):
        moves = board.return_legal_moves(player, dice)
        next_player = 1 if player == 2 else 2
        base_obs = board._return_tesauro_transform(player)

        if len(moves) == 0:
            obs = board._return_tesauro_transform(next_player)
            return self.forward(torch.tensor(obs, dtype=torch.float32)).item(), [], base_obs, obs
        
        saved_positions = list(board.positions) 
        saved_pip = list(board.pip) 
        saved_bos = list(board.bear_off_status)

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            obs = board._return_tesauro_transform(next_player)
            board.positions = list(saved_positions)
            board.pip = list(saved_pip)
            board.bear_off_status = list(saved_bos)
            return self.forward(torch.tensor(obs, dtype=torch.float32)).item(), moves[0], base_obs, obs

        obs_list = [None] * len(moves)
        for i in range(len(moves)):
            board.positions = list(saved_positions)
            board.pip = list(saved_pip) if isinstance(saved_pip, list) else saved_pip
            board.bear_off_status = list(saved_bos) if isinstance(saved_bos, list) else saved_bos
            
            move = moves[i]
            board.execute_move(player, move)
            
            obs_list[i] = board._return_tesauro_transform(next_player)

        board.positions = list(saved_positions)
        board.pip = list(saved_pip) if isinstance(saved_pip, list) else saved_pip
        board.bear_off_status = list(saved_bos) if isinstance(saved_bos, list) else saved_bos

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            evals = self.forward(obs_tensor).squeeze(dim=-1).tolist() #Suggested optimization by Gemini
            
        if player == 1:
            idx = np.argmax(evals)
        else:
            idx = np.argmin(evals)

        return evals[idx], moves[idx], base_obs, obs_list[idx]

    def predict_all(self, board:Logic.Board, dice:tuple[int, int], player:int):
        moves = board.return_legal_moves(player, dice)
        next_player = 1 if player == 2 else 2
        base_obs = board._return_tesauro_transform(player)

        if len(moves) == 0:
            obs = board._return_tesauro_transform(next_player)
            return self.forward(torch.tensor(obs, dtype=torch.float32)).item(), [], base_obs, obs
        
        saved_positions = list(board.positions) 
        saved_pip = list(board.pip) 
        saved_bos = list(board.bear_off_status)

        if len(moves) == 1:
            board.execute_move(player, moves[0])
            obs = board._return_tesauro_transform(next_player)
            board.positions = list(saved_positions)
            board.pip = list(saved_pip)
            board.bear_off_status = list(saved_bos)
            return self.forward(torch.tensor(obs, dtype=torch.float32)).item(), moves[0], base_obs, obs

        obs_list = [None] * len(moves)
        for i in range(len(moves)):
            board.positions = list(saved_positions)
            board.pip = list(saved_pip) if isinstance(saved_pip, list) else saved_pip
            board.bear_off_status = list(saved_bos) if isinstance(saved_bos, list) else saved_bos
            
            move = moves[i]
            board.execute_move(player, move)
            
            obs_list[i] = board._return_tesauro_transform(next_player)

        board.positions = list(saved_positions)
        board.pip = list(saved_pip) if isinstance(saved_pip, list) else saved_pip
        board.bear_off_status = list(saved_bos) if isinstance(saved_bos, list) else saved_bos

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            evals = self.forward(obs_tensor).squeeze(dim=-1).tolist() #Suggested optimization by Gemini
            
        if player == 1:
            ranks = reversed(np.argsort(evals))
        else:
            ranks = np.argsort(evals)

        return evals, moves, ranks

    def run_training_game(self):
        board = Logic.Board()
        roll = Logic.rollDice(first=True)
        player = 1 if roll[0] > roll[1] else 2
        
        traces = {name: torch.zeros_like(param) for name, param in self.named_parameters()}

        while not board.is_game_over():
            val, action, base_obs, next_obs = self.predict(board,roll,player)
            board.execute_move(player,action)
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()
            self.zero_grad()
            v_s = self.forward(torch.tensor(base_obs, dtype=torch.float32))
            v_s.backward()
            with torch.no_grad():
                if board.is_game_over():
                    reward = int(board.get_winner() == 1)
                    td_error = reward - v_s.item()
                else:
                    v_s_next = self.forward(torch.tensor(next_obs, dtype=torch.float32))
                    td_error = v_s_next.item() - v_s.item()


                for name, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        traces[name] = (self.trace_decay * traces[name]) + p.grad
                        p.data += self.learning_rate * td_error * traces[name]

        self.epochs_trained += 1


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


