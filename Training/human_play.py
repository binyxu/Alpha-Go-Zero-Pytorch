from __future__ import print_function
import pickle
from policy_value_net_pytorch import PolicyValueNet
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure, RandPlayer
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import time


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if location.isdecimal():
                location = int(location)
                if location>=0 and location<=44:
                    move = int(location/10) * 5 + location % 10
            else:
                if location=="PASS" or location=="pass" or location=="Pass":
                    move = 25
                else:
                    if isinstance(location, str):  # for python3
                        location = [int(n, 10) for n in location.split(",")]
                        move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    width, height = 5, 5
    model_file = 'best_model_weight.model'
    try:
        board = Board(width=width, height=height)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=500)  # set larger n_playout for better performance

        # best_policy = PolicyValueNetNumpy(width, height, model_file)
        # mcts_player1 = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=500)  # set larger n_playout for better performance

        # mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=500)  # set larger n_playout for better performance

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=2000)

        random_player = RandPlayer()

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        start = time.time()
        game.start_play(mcts_player2, human, start_player=0, is_shown=1)
        end = time.time()
        print(end - start)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
