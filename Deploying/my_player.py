#The following five packages are all build-in packages in Python
import random                    #Usage: give randomness to policy
import time                      #Usage: count time to avoid time out
import pickle                    #Usage: convert file into a byte stream for file i/o
import copy                      #Usage: use deepcopy to copy variables
import os                        #Usage: delete files


#The following package is additional package
import numpy as np               #Usage: mathematical computation


#The following files are dependencies
from read import readInput       #Usage: get input
from write import writeOutput    #Usage: get output


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def relu(X):
    out = np.maximum(X, 0)
    return out


def conv_forward(X, W, b, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.shape
    # theano conv2d flips the filters (rotate 180 degree) first
    # while doing the calculation
    W = W[:, :, ::-1, ::-1]
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_indices(X, h_filter, w_filter,
                           padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = (np.dot(W_col, X_col).T + b).T
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    return out


def fc_forward(X, W, b):
    out = np.dot(X, W) + b
    return out


def get_im2col_indices(x_shape, field_height,
                       field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height,
                                 field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


class PolicyValueNetNumpy():
    """policy-value network in numpy """
    def __init__(self, board_width, board_height, net_params):
        self.board_width = board_width
        self.board_height = board_height
        self.params = net_params

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()

        X = current_state.reshape(-1, 8, self.board_width, self.board_height)
        # first 3 conv layers with ReLu nonlinearity
        for i in [0, 2, 4]:
            X = relu(conv_forward(X, self.params[i], self.params[i+1]))
        # policy head
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.flatten(), self.params[8], self.params[9])
        act_probs = softmax(X_p)
        # value head
        X_v = relu(conv_forward(X, self.params[10],
                                self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # leaf_value = leaf_value.cpu().numpy()
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height + 1)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 5))
        self.height = int(kwargs.get('height', 5))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.players = [1, 2]  # player1 and player2
        self.error = 0

    def init_board(self, cu_pl=0, st_pl=1, available_points=[], current_states={}, rd=0, lm=-1, jl=-1):
        self.current_player = self.players[cu_pl]  # start player
        self.first_player = self.players[st_pl]
        # keep available moves in a list
        self.availables = available_points
        self.states = current_states
        self.round = rd
        self.last_move = lm
        self.jielock = jl

    def move_to_location(self, move):
        """
        5*5 board's moves like:
        0  1  2  3  4
        5  6  7  8  9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24
        and move 11's location is (2,1)
        """
        h = move // 5
        w = move % 5
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*
        Plane 1: My chess piece 1/0
        Plane 2: Opponent's chess piece 1/0
        Plane 3: Opponent's last chess piece 1/0
        Plane 4: Am I the first? 1/0
        Plane 5: Does he pass? 1/0
        Plane 6: If the last move 1/0
        Plane 7: Kill point pieces 1/0
        Plane 8: Death point pieces 1/0
        """

        square_state = np.zeros((8, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            # indicate the last move location
            if self.last_move != 25:
                square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
            for ava_pts in self.availables:
                if self.have_srd(ava_pts, move_curr, move_oppo) == 1:
                    if self.count_qi(ava_pts, move_curr, move_oppo) == 0:
                        square_state[7][ava_pts // self.width, ava_pts % self.height] = 1.0
                    elif self.count_enemy_qi(ava_pts, move_curr, move_oppo) == 0:
                        square_state[6][ava_pts // self.width, ava_pts % self.height] = 1.0
        if self.round % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        if self.last_move == 25:
            square_state[4][:, :] = 1.0  # indicate the colour to play
        if self.round == 23:
            square_state[5][:] = 1.0
        if self.round == 24:
            square_state[5][:] = 10.0

        return square_state[:, ::-1, :]

    def do_move(self, move):
        if move != 25:
            self.last_move = move
            self.round += 1
            self.states[move] = self.current_player

            if self.states:
                moves, players = np.array(list(zip(*self.states.items())))
                move_curr = moves[players == self.current_player]
                move_oppo = moves[players != self.current_player]
            else:
                move_curr = []
                move_oppo = []

            self.availables.remove(move)
            remove_list = self.one_round_remove(move, move_curr, move_oppo)
            self.availables += remove_list
            if len(remove_list) == 1 and move in remove_list:
                self.last_move = 25

            for items in remove_list:
                self.states.pop(items)

            if self.jielock != -1:
                self.availables.append(self.jielock)
                self.jielock = -1

            ji = self.if_jie(move, move_curr, move_oppo)
            if ji != -1:
                self.availables.remove(ji)
                self.jielock = ji
            self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
            )
        else:
            self.round += 1
            if self.jielock != -1:
                self.availables.append(self.jielock)
                self.jielock = -1
            self.current_player = (
                self.players[0] if self.current_player == self.players[1]
                else self.players[1]
            )
            self.last_move = move

    def find_surrounding_pieces(self, move):
        move_list = []
        [last_x, last_y] = self.move_to_location(move)
        if last_y != 0: move_list.append(move - 1)
        if last_y != 4: move_list.append(move + 1)
        if last_x != 0: move_list.append(move - 5)
        if last_x != 4: move_list.append(move + 5)
        return move_list

    def have_srd(self, move, move_cur, move_oppo):
        if move - 5 in move_cur or move - 5 in move_oppo:
            return 1
        if move + 5 in move_cur or move + 5 in move_oppo:
            return 1
        if move - 1 in move_cur or move - 1 in move_oppo:
            return 1
        if move + 1 in move_cur or move + 1 in move_oppo:
            return 1
        return 0

    def find_all_allys(self, move, move_curr):
        ally_list = [move]
        banlist = []
        length = len(move_curr)
        i = 0
        while i < length:
            if i in banlist:
                i += 1
                continue
            for items in self.find_surrounding_pieces(move_curr[i]):
                if items in ally_list:
                    ally_list.append(move_curr[i])
                    banlist.append(i)
                    i = -1
                    break
            i += 1
        ally_list.remove(move)
        if move not in ally_list:
            ally_list.append(move)
        return ally_list

    def find_allys_surrounding(self, ally_list):
        srd_list = []
        for items in ally_list:
            intlist = [element for element in self.find_surrounding_pieces(items) if
                       element not in srd_list + ally_list]
            srd_list = srd_list + intlist
        return srd_list

    def remove_of_one_piece(self, move, move_curr, move_oppo):
        ally_list = self.find_all_allys(move, move_curr)
        srd_list = self.find_allys_surrounding(ally_list)
        if not [element for element in srd_list if element not in move_oppo]:
            return ally_list
        else:
            return []

    def count_qi(self, move, move_curr, move_oppo):
        for pieces in self.find_surrounding_pieces(move):
            if pieces not in move_curr and pieces not in move_oppo:
                return -1
        ally_list = self.find_all_allys(move, move_curr)
        srd_list = self.find_allys_surrounding(ally_list)
        count = 0
        for element in srd_list:
            if element not in move_oppo:
                count += 1
        return count

    def count_enemy_qi(self, move, move_curr, move_oppo):
        tt = 0
        for pieces in self.find_surrounding_pieces(move):
            if pieces in move_oppo:
                if tt == 0:
                    move_currrrrr = [element for element in move_curr]
                    if move not in move_currrrrr:
                        move_currrrrr.append(move)
                    tt = 1
                if self.count_qi(pieces, move_oppo, move_currrrrr) == 0:
                    return 0
        return 1

    def one_round_remove(self, move, move_curr, move_oppo):
        remove_list = []
        tt = [element for element in move_curr]
        if move not in tt:
            tt.append(move)

        for element in self.find_surrounding_pieces(move):
            if element in move_oppo:
                for item in self.remove_of_one_piece(element, move_oppo, tt):
                    if item not in remove_list:
                        remove_list.append(item)

        for element in self.remove_of_one_piece(move, tt, [item for item in move_oppo if item not in remove_list]):
            if element not in remove_list:
                remove_list.append(element)
        return remove_list

    def renew_availables(self, enmlp, removed_list, move_curr, move_oppo):
        removeded = []
        for items in self.availables:
            if items != 25 and self.count_qi(items, move_curr, move_oppo) == 0:
                removelist = self.one_round_remove(items, move_curr, move_oppo)
                if len(removelist) == 1:
                    if removelist[0]==items:
                        removeded.append(items)

        if len(removed_list) == 1:
            rlist = self.one_round_remove(removed_list[0], move_curr, move_oppo)
            if len(rlist) == 1:
                if rlist[0] == enmlp:
                    if removed_list[0] not in removeded:
                        removeded.append(removed_list[0])
                    self.jielock = removed_list[0]

        for items in removeded:
            self.availables.remove(items)

        return 0

    def if_jie(self, move, move_curr, move_oppo):
        remove_list = self.one_round_remove(move, move_curr, move_oppo)

        tt = [element for element in move_curr]
        if move not in tt:
            tt.append(move)

        if len(remove_list) == 1 and remove_list[0] in move_oppo:
            remove_list_2 = self.one_round_remove(remove_list[0],
                                                  [item for item in move_oppo if item not in remove_list],
                                                  tt)
            if len(remove_list_2) == 1 and remove_list_2[0] == move:
                return remove_list[0]
        return -1

    def has_a_winner(self):
        tiemu = 2.5
        if self.round < 24:
            return False, -1
        else:
            player_1_score = (len([k for k, v in self.states.items() if v == 1]))
            player_2_score = (len([k for k, v in self.states.items() if v == 2]))
            if self.first_player == 1:
                player_2_score += tiemu
            else:
                player_1_score += tiemu

            if (player_1_score == player_2_score):
                return True, -1
            elif player_1_score > player_2_score:
                return True, 1
            else:
                return True, 2

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class RandomPlayer():
    def __init__(self):
        self.type = 'random'

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    possible_placements.append((i, j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)


if __name__ == "__main__":
    start = time.time()
    N = 5
    piece_type, previous_board, board = readInput(N)

    last_move = -1
    move_curr = []
    move_oppo = []
    count_pieces = 0
    available_pts = [25]
    removed_pts = []
    states = {}

    for i in range(N):
        for j in range(N):
            if board[i][j] - previous_board[i][j] == 3 - piece_type:
                last_move = i * N + j
            elif previous_board[i][j] - board[i][j] == piece_type:
                removed_pts.append(i * N + j)
            if board[i][j] == piece_type:
                move_curr.append(i * N + j)
                count_pieces += 1
            elif board[i][j] == 3 - piece_type:
                move_oppo.append(i * N + j)
                count_pieces += 1
            elif board[i][j] == 0:
                available_pts.append(i * N + j)

    if count_pieces == 0:
        round = 0
        with open('round.pkl', 'wb') as f:
            pickle.dump(round, f)

    elif count_pieces == 1:
        round = 1
        with open('round.pkl', 'wb') as f:
            pickle.dump(round, f)

    else:
        with open('round.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            round = pickle.load(f)
        round = round + 2
        if round == 23 or round == 22:
            os.remove('round.pkl')
        else:
            with open('round.pkl', 'wb') as f:
                pickle.dump(round, f)

    if round % 2 == 0:
        first_player = piece_type
    else:
        first_player = 3 - piece_type

    for items in move_curr:
        states[items] = piece_type
    for items in move_oppo:
        states[items] = 3 - piece_type

    my_board = Board(width=N, height=N)
    my_board.init_board(piece_type - 1, first_player - 1, available_pts, states, round, last_move, -1)
    my_board.renew_availables(last_move, removed_pts, move_curr, move_oppo)

    model_file = 'best_model_weight_numpy.model'
    try:
        policy_param = pickle.load(open(model_file, 'rb'))
    except:
        policy_param = pickle.load(open(model_file, 'rb'),
                                    encoding='bytes')  # To support python3
    best_policy = PolicyValueNetNumpy(N, N, policy_param)
    mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
                              c_puct=N,
                              n_playout=500)

    player_in_turn = mcts_player2
    move = player_in_turn.get_action(my_board)

    if move >= 0 and move <= 24:
        action = (int(move / 5), int(move % 5))
    else:
        action = 'PASS'

    removelist = my_board.one_round_remove(move, move_curr, move_oppo)
    if len(removelist) == 1:
        if removelist[0]==move:
            action = 'PASS'
    print(action)
    writeOutput(action)
    end = time.time()
