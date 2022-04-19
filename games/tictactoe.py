import datetime
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from AI import *
from games.abstract_game import AbstractGame
from games.tictactoe_dic.Empty import Empty
from games.tictactoe_dic.Piece import Piece
from tkinter import *
from games.tictactoe_dic.GUI import GUI
from Launcher import MuZero


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 3,
                                  3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(9))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 9  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
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
        self.env = TicTacToe()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

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

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        '''
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                        choice in self.legal_actions()
                        and 1 <= row
                        and 1 <= col
                        and row <= 3
                        and col <= 3
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        '''
        self.env.gui.choice = -1
        while self.env.gui.choice == -1:
            pass
        return self.env.gui.choice

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
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


class TicTacToe:
    def __init__(self):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.player

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)  # ai have put = 1 others 0
        board_player2 = numpy.where(self.board == -1, 1, 0)  # human have put = 1 others 0
        board_to_play = numpy.full((3, 3), self.player)  # if this is human = 1 or -1
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
                self.board[0, 0] == self.player
                and self.board[1, 1] == self.player
                and self.board[2, 2] == self.player
        ):
            return True
        if (
                self.board[2, 0] == self.player
                and self.board[1, 1] == self.player
                and self.board[0, 2] == self.player
        ):
            return True

        return False

    def expert_action(self):
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = numpy.where(board[i, :] == 0)[0][0]
                action = numpy.ravel_multi_index(
                    (numpy.array([i]), numpy.array([ind])), (3, 3)
                )[0]
                if self.player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = numpy.where(board[:, i] == 0)[0][0]
                action = numpy.ravel_multi_index(
                    (numpy.array([ind]), numpy.array([i])), (3, 3)
                )[0]
                if self.player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = numpy.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = numpy.where(diag == 0)[0][0]
            action = numpy.ravel_multi_index(
                (numpy.array([ind]), numpy.array([ind])), (3, 3)
            )[0]
            if self.player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = numpy.where(anti_diag == 0)[0][0]
            action = numpy.ravel_multi_index(
                (numpy.array([ind]), numpy.array([2 - ind])), (3, 3)
            )[0]
            if self.player * sum(anti_diag) > 0:
                return action

        return action

    def render(self):
        print(self.board[::-1])


class tictactoe_gui:
    def __init__(self, root):
        self.root = root
        self.name = "tictactoe"
        self.play = False
        self.states = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(Empty(0, ""))
            self.states.append(row)

    def train(self):
        MuZero("tictactoe").train()

    def option(self, img):
        self.root.clear_all()

        Label(self.root, text="Tic Tac Toe", font=("Times New Roman", "25"), bg="#FFFFFF").place(x=40, y=20)

        Canvas(self.root, bg="#D8D8D8", height=5, width=700).place(x=30, y=70)

        Canvas(self.root, bg="#D8D8D8", height=470, width=123).place(x=30, y=100)

        Canvas(self.root, bg="#D8D8D8", height=470, width=552).place(x=180, y=100)

        Label(self.root,
              text="游戏名",
              font=("microsoft yahei", "12"),
              wraplength=450,
              bg="#D8D8D8",
              justify="left").place(x=200, y=120)

        v_1 = StringVar()
        v_1.set(MuZeroConfig().network)

        Entry(self.root,
              textvariable=v_1,
              width=15,
              borderwidth=0).place(x=270, y=125)

        Label(self.root, image=img, width=100, height=100, bd=2, relief="solid").place(x=40, y=110)

    def detail(self, img):
        self.root.clear_all()

        Label(self.root, text="Tic Tac Toe", font=("Times New Roman", "25"), bg="#FFFFFF").place(x=40, y=20)

        Canvas(self.root, bg="#D8D8D8", height=5, width=700).place(x=30, y=70)

        Canvas(self.root, bg="#D8D8D8", height=470, width=700).place(x=30, y=100)

        Label(self.root, image=img, width=200, height=200, bd=2, relief="solid").place(x=50, y=120)

        Label(self.root,
              text="简介",
              font=("microsoft yahei", "15", "bold"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=113)

        Label(self.root,
              text="是黑白棋的一种。三子棋是一种民间传统游戏，又叫九宫棋、圈圈叉叉、一条龙、井字棋等。将正方形对角线连起来，相对两边依次摆上"
                   "三个双方棋子。",
              font=("microsoft yahei", "12"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=140)

        Label(self.root,
              text="规则",
              font=("microsoft yahei", "15", "bold"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=210)

        Label(self.root,
              text="只要将自己的三个棋子走成一条线，对方就算输了。如果两个人都掌握了技巧，那么一般来说就是平棋。一般来说，第二步下在中间最有"
                   "利（因为第一步不能够下在中间），下在角上次之，下在边上再次之。",
              font=("microsoft yahei", "12"), wraplength=450, bg="#D8D8D8", justify="left").place(x=265, y=237)

        checkpoint_path = "results/tictactoe/model.checkpoint"
        checkpoint_path = pathlib.Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        replay_buffer_path = "results/tictactoe/replay_buffer.pkl"

        replay_buffer_path = pathlib.Path(replay_buffer_path)
        with open(replay_buffer_path, "rb") as f:
            replay_buffer_infos = pickle.load(f)
            replay_buffer = replay_buffer_infos["buffer"]

        # game_priority
        x = []
        y = []

        total_win = 0
        horizontal_line = []
        for i in range(checkpoint["num_played_games"]):
            x.append(i)
            steps = len(replay_buffer[i].reward_history)
            win = 1 if replay_buffer[i].to_play_history[steps - 1] == 1 else -1
            total_win += win
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
        self.game_history.observation_history.append(numpy.zeros([3, 3]))
        self.game_history.reward_history.append(0)
        self.game_history.to_play_history.append(0)
        self.ai.load_model("results/tictactoe/model.checkpoint", "results/tictactoe/replay_buffer.pkl")

        self.root.clear_all()
        self.canvas = GUI(self.root)
        self.root.unbind_all("<Button>")
        self.root.bind_all("<Button>", self.button_press)
        self.root.bind_all("<Key>", self.key_press)

        self.testing = True

        while self.testing:
            self.canvas.render(self.states)
            if not self.play:
                action, root = self.ai.get_action(self.game_history)
                self.new_piece(action // 3, action % 3, 'white')
                self.ai.update_history(action, root, self.game_history,
                                       1 if self.judge_all(action // 3, action % 3) else 0, 1)
                self.play = True

        self.root.clear_all()
        self.root.setObjects()

    def get_legal_actions(self):
        actions = []
        for i in range(3):
            for j in range(3):
                if type(self.states[i][j]) == Empty:
                    actions.append(i * 3 + j)

        return actions

    def get_observation(self):
        board_A = numpy.zeros([3, 3])
        board_B = numpy.zeros([3, 3])
        board_C = numpy.ones([3, 3])

        if not self.play:
            board_C = -board_C

        for i in range(3):
            for j in range(3):
                if type(self.states[i][j]) == Piece:
                    if self.states[i][j].color == 'black':
                        board_A[i][j] = 1
                    else:
                        board_B[i][j] = 1

        return board_A, board_B, board_C

    def button_press(self, e):
        x = int(3 * e.x / self.canvas.width / self.canvas.unit)
        y = int(3 * e.y / self.canvas.height / self.canvas.unit)
        if self.play and type(self.states[x][y]) is Empty:
            self.new_piece(x, y, 'black')
            action, root = self.ai.get_action(self.game_history)
            self.ai.update_history((x - 1) * 3 + y - 1, root, self.game_history, 1 if self.judge_all(x, y) else 0, 0)
            self.play = False

    def new_piece(self, x, y, color):
        self.states[x][y] = Piece(15, color)
        if self.judge_all(x, y):
            print(color, "win.")

    def judge_all(self, x0, y0):
        for (i, j) in ((1, 0), (0, 1), (1, -1), (1, 1)):
            if self.judge(x0, y0, i, j):
                return True

        return False

    def judge(self, x0, y0, dx, dy):
        total = 0
        for i in range(1, 3):
            if x0 + dx * i > 2 or y0 + dy * i > 2 or x0 + dx * i < 0 or y0 + dy * i < 0 or \
                    self.states[x0 + dx * i][y0 + dy * i].color != self.states[x0][y0].color:
                break
            total += 1
        for i in range(1, 3 - total):
            if x0 - dx * i > 2 or y0 - dy * i > 2 or x0 - dx * i < 0 or y0 - dy * i < 0 or \
                    self.states[x0 - dx * i][y0 - dy * i].color != self.states[x0][y0].color:
                break
            total += 1

        return total > 1

    def key_press(self, e):
        if e.keysym == "Escape":
            self.testing = False
