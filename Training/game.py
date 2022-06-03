from __future__ import print_function

import time

import numpy as np
import copy


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

    def init_board(self, start_player=0):
        self.current_player = self.players[start_player]  # start player
        self.first_player = self.players[start_player]
        # keep available moves in a list
        self.availables = list(range(self.width * self.height + 1))
        self.states = {}
        self.round = 0
        self.last_move = -1
        self.jielock = -1

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
        if move-5 in move_cur or move-5 in move_oppo:
            return 1
        if move+5 in move_cur or move+5 in move_oppo:
            return 1
        if move-1 in move_cur or move-1 in move_oppo:
            return 1
        if move+1 in move_cur or move+1 in move_oppo:
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


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        if board.last_move == 25:
            print("PASS")
            print('\r\n\r\n')
        else:
            print("Player", player1, "with X".rjust(3))
            print("Player", player2, "with O".rjust(3))
            print()
            for x in range(width):
                print("{0:8}".format(x), end='')
            print('\r\n')
            for i in range(height - 1, -1, -1):
                print("{0:4d}".format(i), end='')
                for j in range(width):
                    loc = i * width + j
                    p = board.states.get(loc, -1)
                    if p == player1:
                        print('X'.center(8), end='')
                    elif p == player2:
                        print('O'.center(8), end='')
                    else:
                        print('_'.center(8), end='')
                print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    print(winner)
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
