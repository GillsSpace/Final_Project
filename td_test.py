import torch
import torchinfo
import Logic
import numpy as np
import time

class TDModel(torch.nn.Module):
    def __init__(self):
        super(TDModel, self).__init__()

        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(198, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

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

EPOCHS = 1000
LAMBDA = 0.8
ALPHA = 0.001


print(torchinfo.summary(TDModel(), input_size=(1, 198)))

model = TDModel()
print("predictions for opening moves:")
board = Logic.Board()
for roll in Logic.ROLLS:
    if roll[0] == roll[1]:
        continue
    val, action, _, _ = model.predict(board,roll,1)
    print(f"Roll: {roll} --> Move: {action} ({val})")

st = time.time()
for epoch in range(EPOCHS):
    board = Logic.Board()
    roll = Logic.rollDice(first=True)
    player = 1 if roll[0] > roll[1] else 2

    traces = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    while not board.is_game_over():
        val, action, base_obs, next_obs = model.predict(board,roll,player)
        board.execute_move(player,action)
        player = 1 if player == 2 else 2
        roll = Logic.rollDice()
        model.zero_grad()
        v_s = model.forward(torch.tensor(base_obs, dtype=torch.float32))
        v_s.backward()
        with torch.no_grad():
            if board.is_game_over():
                reward = int(board.get_winner() == 1)
                td_error = reward - v_s.item()
                if epoch % 2000 == 0:
                    et = time.time()
                    print(f'Training for {et-st} seconds')
                    print(f'Final Board State of epoch {epoch}:')
                    board.render_terminal()
                    print(f'Player {board.get_winner()} won with move: {action} ({val})')
            else:
                v_s_next = model.forward(torch.tensor(next_obs, dtype=torch.float32))
                td_error = v_s_next.item() - v_s.item()


            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    traces[name] = (LAMBDA * traces[name]) + p.grad
                    p.data += ALPHA * td_error * traces[name]

    


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


