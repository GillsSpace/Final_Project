import Logic
import time
import random

board = Logic.Board()
board.render_terminal()

print(f'Legal moves for player 1 with dice (2,3):', end=' ')
print(*board.return_legal_moves(1, (2, 3)), sep=' -- ')

print(f'Legal moves for player 2 with dice (6,6):', end=' ')
print(*board.return_legal_moves(2, (6, 6)), sep=' -- ')

st = time.time()
for _ in range(100):
    board = Logic.Board()
    player = random.choice([1, 2])
    while not board.is_game_over():
        dice = Logic.rollDice()
        legal_moves = board.return_legal_moves(player, dice)
        if legal_moves:
            move = random.choice(legal_moves)
            board.execute_move(player, move)
        player = 1 if player == 2 else 2
        board.render_terminal()
et = time.time()
print(f"Average time per game: {(et - st) / 100:.4f} seconds")

