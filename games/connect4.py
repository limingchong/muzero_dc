import datetime
import pathlib

from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from games.connect4_dic.Piece import Piece
from games.connect4_dic.GUI import GUI
from games.connect4_dic.Empty import Empty
from AI import *
from games.abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 6,
                                  7)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(7))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Connect4()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"


class Connect4:
    def __init__(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((6, 7), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((6, 7), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(7):
            if self.board[5][i] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal check
        for i in range(4):
            for j in range(6):
                if (
                        self.board[j][i] == self.player
                        and self.board[j][i + 1] == self.player
                        and self.board[j][i + 2] == self.player
                        and self.board[j][i + 3] == self.player
                ):
                    return True

        # Vertical check
        for i in range(7):
            for j in range(3):
                if (
                        self.board[j][i] == self.player
                        and self.board[j + 1][i] == self.player
                        and self.board[j + 2][i] == self.player
                        and self.board[j + 3][i] == self.player
                ):
                    return True

        # Positive diagonal check
        for i in range(4):
            for j in range(3):
                if (
                        self.board[j][i] == self.player
                        and self.board[j + 1][i + 1] == self.player
                        and self.board[j + 2][i + 2] == self.player
                        and self.board[j + 3][i + 3] == self.player
                ):
                    return True

        # Negative diagonal check
        for i in range(4):
            for j in range(3, 6):
                if (
                        self.board[j][i] == self.player
                        and self.board[j - 1][i + 1] == self.player
                        and self.board[j - 2][i + 2] == self.player
                        and self.board[j - 3][i + 3] == self.player
                ):
                    return True

        return False

    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        for k in range(3):
            for l in range(4):
                sub_board = board[k: k + 4, l: l + 4]
                # Horizontal and vertical checks
                for i in range(4):
                    if abs(sum(sub_board[i, :])) == 3:
                        ind = numpy.where(sub_board[i, :] == 0)[0][0]
                        if numpy.count_nonzero(board[:, ind + l]) == i + k:
                            action = ind + l
                            if self.player * sum(sub_board[i, :]) > 0:
                                return action

                    if abs(sum(sub_board[:, i])) == 3:
                        action = i + l
                        if self.player * sum(sub_board[:, i]) > 0:
                            return action
                # Diagonal checks
                diag = sub_board.diagonal()
                anti_diag = numpy.fliplr(sub_board).diagonal()
                if abs(sum(diag)) == 3:
                    ind = numpy.where(diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, ind + l]) == ind + k:
                        action = ind + l
                        if self.player * sum(diag) > 0:
                            return action

                if abs(sum(anti_diag)) == 3:
                    ind = numpy.where(anti_diag == 0)[0][0]
                    if numpy.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                        action = 3 - ind + l
                        if self.player * sum(anti_diag) > 0:
                            return action

        return action

    def render(self):
        print(self.board[::-1])


class connect4_gui:
    def __init__(self, root):
        self.root = root
        self.name = "connect4"
        self.play = False
        self.winfor = 4
        self.board_width = 6
        self.board_height = 7

    def train(self):
        pass

    def detail(self, img):
        self.root.clear_all()

        Label(self.root, text="Connect Four", font=("Times New Roman", "25"), bg="#FFFFFF").place(x=40, y=20)

        Canvas(self.root, bg="#D8D8D8", height=5, width=700).place(x=30, y=70)

        Canvas(self.root, bg="#D8D8D8", height=470, width=700).place(x=30, y=100)

        Label(self.root, image=img, width=200, height=200, bd=2, relief="solid").place(x=50, y=120)

        Label(self.root,
              text="简介",
              font=("microsoft yahei", "15", "bold"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=113)

        Label(self.root,
              text="Connect Four四子棋智力棋盘游戏是一种供两人对弈的棋类游戏。圣诞长假即将到来，可能今年更多人会呆家里而不是出游。"
                   "即使呆家里也不要忘记玩一些好玩的游戏增添乐趣。",
              font=("microsoft yahei", "12"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=140)

        Label(self.root,
              text="规则",
              font=("microsoft yahei", "15", "bold"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=210)

        Label(self.root,
              text="在棋盘中，任何一方先令自己四只棋子在横，竖或斜方向联成一条直线，即可获胜。类似我们平时玩的五子棋。（适合六岁以上和大人）",
              font=("microsoft yahei", "12"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=237)

        checkpoint_path = "results/connect4/model.checkpoint"
        checkpoint_path = pathlib.Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        replay_buffer_path = "results/connect4/replay_buffer.pkl"

        replay_buffer_path = pathlib.Path(replay_buffer_path)
        with open(replay_buffer_path, "rb") as f:
            replay_buffer_infos = pickle.load(f)
            replay_buffer = replay_buffer_infos["buffer"]

        # game_priority
        x = []
        y = []

        total_win = 0
        horizontal_line = []
        for i in range(checkpoint["num_played_games"] - len(replay_buffer) + 10 , checkpoint["num_played_games"]):
            x.append(i)
            steps = len(replay_buffer[i].reward_history)
            if replay_buffer[i].to_play_history[steps - 1] == 1:
                total_win += 1
            y.append(total_win / (i + 1))
            horizontal_line.append(0.5)

        string = "训练局数：" + str(checkpoint["num_played_games"]) + "\n" \
                                                                 "平均步数：" + str(
            checkpoint["num_played_steps"] / checkpoint["num_played_games"]) + "\n" \
                                                                "平均损失：" + str(checkpoint["reward_loss"])

        Label(self.root, text=string, font=("microsoft yahei", "12", "bold"), wraplength=450, bg="#D8D8D8",
              justify="left").place(x=40, y=350)
        Button(self.root, text="重新训练", bg="#D8D8D8", borderwidth=2, fg="black",
               font=("microsoft yahei", "12", "bold")).place(x=40, y=450)
        Button(self.root, text="关于我们", bg="#D8D8D8", borderwidth=2, fg="black",
               font=("microsoft yahei", "12", "bold")).place(x=40, y=500)

        f = pyplot.figure()

        pyplot.plot(x, y, ls='-', lw=1, label='win rate', color='purple')
        pyplot.plot(x, horizontal_line, ls='-', lw=2, label='50%', color='red')
        pyplot.legend()
        pyplot.xlabel("epoch")
        pyplot.ylabel("rate")

        plot_show = FigureCanvasTkAgg(f, self.root)
        plot_show.get_tk_widget().pack(side=BOTTOM, expand=True)
        plot_show.get_tk_widget().place(x=400, y=350)
        plot_show.get_tk_widget().config(width=300, height=200)

    def test(self):
        self.ai = AI(self, MuZeroConfig(), 666)
        self.game_history = GameHistory()
        self.game_history.action_history.append(0)
        self.game_history.observation_history.append(numpy.zeros([6, 7]))
        self.game_history.reward_history.append(0)
        self.game_history.to_play_history.append(0)
        self.ai.load_model("results/connect4/model.checkpoint", "results/connect4/replay_buffer.pkl")

        self.states = []

        for i in range(6):
            row = []
            for j in range(7):
                row.append(Empty(0, ""))
            self.states.append(row)

        self.canvas = GUI(self.root)

        self.root.unbind_all("<Button>")
        self.root.bind_all("<Button>", self.button_press)
        self.root.bind_all("<Key>", self.key_press)

        self.testing = True
        while self.testing:
            self.canvas.render(self.states)
            if not self.play:
                action, root = self.ai.get_action(self.game_history)
                for x in range(6):
                    if type(self.states[x][action]) == Empty:
                        self.new_piece(x, action, 'red')
                        self.ai.update_history(action, root, self.game_history,
                                               1 if self.judge_all(x, action) else 0, 1)
                        break
                self.play = True

        self.root.clear_all()
        self.root.setObjects()

    def get_observation(self):
        board_A = numpy.zeros([6, 7])
        board_B = numpy.zeros([6, 7])
        board_C = numpy.ones([6, 7])

        if not self.play:
            board_C = -board_C

        for i in range(6):
            for j in range(7):
                if type(self.states[i][j]) == Piece:
                    if self.states[i][j].color == 'yellow':
                        board_A[i][j] = 1
                    else:
                        board_B[i][j] = 1

        return board_A, board_B, board_C

    def get_legal_actions(self):
        actions = []
        for i in range(6):
            if type(self.states[i][6]) == Empty:
                actions.append(i)

        return actions

    def button_press(self, e):
        y = int(6 * e.y / self.canvas.height / self.canvas.unit)
        for x in range(6):
            if type(self.states[x][y]) == Empty and self.play:
                self.new_piece(x, y, 'yellow')
                #action, root = self.ai.get_action(self.game_history)
                #self.ai.update_history(y, root, self.game_history, 1 if self.judge_all(x, y) else 0, 0)
                self.play = False
                break


    def new_piece(self, x, y, color):
        self.states[x][y] = Piece(10, color)
        if self.judge_all(x, y):
            print(color, "win.")

    def judge_all(self, x0, y0):
        for (i, j) in ((1, 0), (0, 1), (1, -1), (1, 1)):
            if self.judge(x0, y0, i, j):
                return True

        return False

    def judge(self, x0, y0, dx, dy):
        total = 0
        for i in range(1, 4):
            if x0 + dx * i > 5 or y0 + dy * i > 6 or x0 + dx * i < 0 or y0 + dy * i < 0 or \
                    self.states[x0 + dx * i][y0 + dy * i].color != self.states[x0][y0].color:
                break
            total += 1
        for i in range(1, 4 - total):
            if x0 - dx * i > 5 or y0 - dy * i > 6 or x0 - dx * i < 0 or y0 - dy * i < 0 or \
                    self.states[x0 - dx * i][y0 - dy * i].color != self.states[x0][y0].color:
                break
            total += 1

        return total > 2

    def key_press(self, e):
        if e.keysym == "Escape":
            self.testing = False
