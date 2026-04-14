import gnubg_nn as gnubg
import Logic
import Models

POSITION_SET_A = {
    'A1': [-2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    'A2': [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    'A3': [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2, -2, -2, -2, -2, 2, 0, 0, 0],
}

def run_exhibition_game_terminal(model):
    model = Models.Model_Loader.load_model(model)
    board = Logic.Board()
    roll = Logic.rollDice(True)
    player = 1 if roll[0] > roll[1] else 2

    print(f"Starting Game...")
    board.render_terminal()
    print()
    print(f"Player {player} won opening roll with {roll}")
    while not board.is_game_over():
        val, action, _, _ = model.predict(board,roll,player)
        print(f"Player {player} with roll {roll} makes move {action} - ({val})...")
        board.execute_move(player,action)
        board.render_terminal()
        print()
        player = 1 if player == 2 else 2
        roll = Logic.rollDice()
        
def play_in_terminal(model):
    #P1 = Human, P2 = Bot
    model = Models.Model_Loader.load_model(model)
    board = Logic.Board()
    roll = Logic.rollDice(True)
    player = 1 if roll[0] > roll[1] else 2
    print(f"Starting Game...")
    board.render_terminal()
    print(f"Player {player} won opening roll with {roll}")
    print()
    while not board.is_game_over():
        if player == 2:
            val, action, _, _ = model.predict(board,roll,player)
            print(f"Player {player} with roll {roll} makes move {action} - ({val})...")
            board.execute_move(player,action)
            board.render_terminal()
            print()
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()
        else:
            possible_actions = board.return_legal_moves(1,roll)
            print(f"Roll is {roll}, please select move from choices below:")
            if len(possible_actions) > 0:
                for i in range(len(possible_actions)):
                    print(f'{i+1} --> {possible_actions[i]}')
                choice = int(input('Selected Move: ')) - 1
                action = possible_actions[choice]
                board.execute_move(player,action)
            else:
                _ = input('No Possible Moves. Press ENTER to continue.')
            board.render_terminal()
            print()
            player = 1 if player == 2 else 2
            roll = Logic.rollDice()


if __name__ == "__main__":
    MODEL = 'v1_001.pickle'
    play_in_terminal(MODEL)