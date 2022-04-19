from AI import *

checkpoint_path = "results/tictactoe/model.checkpoint"
checkpoint_path = pathlib.Path(checkpoint_path)
checkpoint = torch.load(checkpoint_path)
replay_buffer_path = "results/tictactoe/replay_buffer.pkl"

replay_buffer_path = pathlib.Path(replay_buffer_path)
with open(replay_buffer_path, "rb") as f:
    replay_buffer_infos = pickle.load(f)
    replay_buffer = replay_buffer_infos["buffer"]

n = checkpoint["num_played_games"]
x = []
y = []
total_win = 0
horizontal_line = []

print(range(checkpoint["num_played_games"] - len(replay_buffer) + 7, checkpoint["num_played_games"]))
for i in range(checkpoint["num_played_games"] - len(replay_buffer) + 7, checkpoint["num_played_games"]):
    x.append(i)
    steps = len(replay_buffer[i].reward_history)
    win = 1 if replay_buffer[i].to_play_history[steps - 1] == 1 else -1
    total_win += win
    y.append(total_win / (i + 1))
    horizontal_line.append(0.5)