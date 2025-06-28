import chess
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
CASTLING_INDICES = {'K': 12, 'Q': 13, 'k': 14, 'q': 15}

def fen_to_tensor_perspective(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    tensor = np.zeros((17, 8, 8), dtype=np.float32)

    if board.turn == chess.WHITE:
        piece_to_plane = PIECE_TO_PLANE
        flip = False
    else:
        piece_to_plane = {
            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5
        }
        flip = True

    parts = fen.split()
    rows = parts[0].split('/')
    for r, row in enumerate(rows):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                plane = piece_to_plane[c]
                tensor[plane, r, f] = 1.0
                f += 1

    if 'K' in parts[2]: tensor[12, :, :] = 1.0
    if 'Q' in parts[2]: tensor[13, :, :] = 1.0
    if 'k' in parts[2]: tensor[14, :, :] = 1.0
    if 'q' in parts[2]: tensor[15, :, :] = 1.0

    if board.ep_square is not None:
        ep_rank = 7 - (board.ep_square // 8)
        ep_file = board.ep_square % 8
        tensor[16, ep_rank, ep_file] = 1.0

    if flip:
        tensor = np.flip(tensor, axis=(1, 2)).copy()

    return torch.tensor(tensor, dtype=torch.float32)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ChessPolicyValueNet(nn.Module):
    def __init__(self, input_planes, policy_size, num_blocks=12, channels=128):
        super().__init__()
        self.conv_in = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_size)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        p = self.relu(self.policy_bn(self.policy_conv(x))).view(x.size(0), -1)
        p = self.policy_fc(p)

        v = self.relu(self.value_bn(self.value_conv(x))).view(x.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = self.value_fc2(v)

        return p, v

with open("move_to_idx.json", "r") as f:
    move_to_idx = json.load(f)
idx_to_move = {int(i): m for m, i in move_to_idx.items()}

model = ChessPolicyValueNet(input_planes=17, policy_size=len(move_to_idx))
model.load_state_dict(torch.load("model.pt", map_location="cpu")['model_state_dict'])
model.eval()

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

def expand_node(node, top_k=10):
    legal_moves = list(node.board.legal_moves)
    tensor = fen_to_tensor_perspective(node.board.fen()).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(tensor)
        policy = torch.softmax(logits[0], dim=0)

    move_priors = []
    for move in legal_moves:
        uci = move.uci()
        if uci in move_to_idx:
            prior = policy[move_to_idx[uci]].item()
            move_priors.append((move, prior))

    # Keep only the top K moves by prior
    move_priors = sorted(move_priors, key=lambda x: x[1], reverse=True)[:top_k]

    for move, prior in move_priors:
        board_copy = node.board.copy()
        board_copy.push(move)
        node.children[move.uci()] = MCTSNode(board_copy, parent=node, move=move, prior=prior)

    return value[0]

def select_child(node, c_puct=1.0):
    total_visits = sum(child.visits for child in node.children.values()) + 1
    best_score = -float('inf')
    best_child = None

    for child in node.children.values():
        u = child.value() + c_puct * child.prior * (total_visits ** 0.5 / (1 + child.visits))
        if u > best_score:
            best_score = u
            best_child = child

    return best_child

def backpropagate(node, value):
    current = node
    win_prob = torch.softmax(value, dim=0)[0].item()
    while current:
        current.visits += 1
        current.value_sum += win_prob
        current = current.parent

def run_mcts(root_board, simulations=1000, top_k=10):
    root = MCTSNode(root_board)
    expand_node(root, top_k=top_k)

    for _ in range(simulations):
        node = root
        while node.children:
            node = select_child(node)

        if node.board.is_game_over():
            outcome = node.board.result()
            if outcome == "1-0":
                value = torch.tensor([1.0, 0.0, 0.0])
            elif outcome == "0-1":
                value = torch.tensor([0.0, 0.0, 1.0])
            else:
                value = torch.tensor([0.0, 1.0, 0.0])
        else:
            value = expand_node(node, top_k=top_k)

        backpropagate(node, value)

    best = max(root.children.values(), key=lambda c: c.visits)
    return best.move.uci()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start-game", methods=["GET"])
def start_game():
    board = chess.Board()
    computer_color = random.choice(['white', 'black'])
    move = None

    if computer_color == 'white':
        move = run_mcts(board, simulations=100, top_k=10)
        board.push(chess.Move.from_uci(move))

    return jsonify({
        "fen": board.fen(),
        "computer_color": computer_color,
        "move": move
    })

@app.route("/model-move", methods=["POST"])
def model_move():
    data = request.get_json()
    fen = data["fen"]
    board = chess.Board(fen)

    if board.is_game_over():
        return jsonify({"error": "Game is over."})

    best_move = run_mcts(board, simulations=100, top_k=10)
    board.push(chess.Move.from_uci(best_move))

    return jsonify({
        "fen": board.fen(),
        "move": best_move
    })

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)

