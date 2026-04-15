import torch
import torchinfo
import Logic
import numpy as np
import time
import pickle

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
    def load_model(filename:str) -> object:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model


