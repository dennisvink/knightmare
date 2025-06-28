import os
import chess
import torch
import numpy as np
from tqdm import tqdm
import random

SHARD_SIZE = 250_000
VAL_FRACTION = 0.1
MOVE_BUCKETS = [
    (0, 10), (10, 20), (20, 30),
    (30, 40), (40, 50), (50, 70), (70, 10_000)
]
BUCKET_NAMES = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-70", "71+"]
ALL_BUCKETS = BUCKET_NAMES + ["mating"]
NUM_INPUT_PLANES = 17

def fen_to_tensor_perspective(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    tensor = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)

    if board.turn == chess.WHITE:
        piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        f_castling, e_castling = (12, 13), (14, 15)
        flip_board = False
    else:
        piece_to_plane = {
            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5
        }
        f_castling, e_castling = (12, 13), (14, 15)
        flip_board = True

    rows = fen.split()[0].split('/')
    for rank_index, row in enumerate(rows):
        file_index = 0
        for char in row:
            if char.isdigit():
                file_index += int(char)
            else:
                if char in piece_to_plane:
                    tensor[piece_to_plane[char], rank_index, file_index] = 1.0
                file_index += 1

    castling_str = fen.split()[2]
    if castling_str != "-":
        if ('K' in castling_str and board.turn == chess.WHITE) or ('k' in castling_str and board.turn == chess.BLACK):
            tensor[f_castling[0], :, :] = 1.0
        if ('Q' in castling_str and board.turn == chess.WHITE) or ('q' in castling_str and board.turn == chess.BLACK):
            tensor[f_castling[1], :, :] = 1.0
        if ('K' in castling_str and board.turn == chess.BLACK) or ('k' in castling_str and board.turn == chess.WHITE):
            tensor[e_castling[0], :, :] = 1.0
        if ('Q' in castling_str and board.turn == chess.BLACK) or ('q' in castling_str and board.turn == chess.WHITE):
            tensor[e_castling[1], :, :] = 1.0

    if board.ep_square is not None:
        rank = 7 - (board.ep_square // 8)
        file = board.ep_square % 8
        tensor[16, rank, file] = 1.0

    if flip_board:
        tensor = np.flip(tensor, axis=(1, 2)).copy()

    return torch.tensor(tensor, dtype=torch.float32)

def load_game_data(fen_path, move_path):
    result = None
    fens, moves = [], []
    with open(fen_path, "r") as f_fen, open(move_path, "r") as f_move:
        first_fen_line = f_fen.readline().strip()
        first_move_line = f_move.readline().strip()
        if first_fen_line.startswith("# Result") and first_move_line.startswith("# Result"):
            result = first_fen_line.split()[-1]
        else:
            f_fen.seek(0)
            f_move.seek(0)
        for fen_line, move_line in zip(f_fen, f_move):
            fen, move = fen_line.strip(), move_line.strip()
            if fen and move:
                fens.append(fen)
                moves.append(move)
    assert len(fens) == len(moves)
    return result, fens, moves

def get_buckets_for_position(idx, total_moves):
    buckets = []
    if total_moves - idx <= 10:
        buckets.append("mating")
    for (start, end), name in zip(MOVE_BUCKETS, BUCKET_NAMES):
        if start <= idx < end:
            buckets.append(name)
            break
    return buckets

def generate_samples(fen_dir, move_dir, val_files_set):
    game_files = sorted(f for f in os.listdir(fen_dir) if f.endswith(".txt"))
    for filename in tqdm(game_files, desc="Streaming"):
        fen_path = os.path.join(fen_dir, filename)
        move_path = os.path.join(move_dir, filename.replace(".txt", ".moves.txt"))
        try:
            result, fens, moves = load_game_data(fen_path, move_path)
        except Exception as e:
            print(f"⚠️ {filename}: {e}")
            continue
        if result is None:
            continue
        winner = "white" if result == "1-0" else "black" if result == "0-1" else "draw"
        is_val = filename in val_files_set
        for idx, (fen, move) in enumerate(zip(fens, moves)):
            if move == "none":
                continue
            try:
                tensor = fen_to_tensor_perspective(fen)
            except Exception as e:
                print(f"⚠️ Tensor conversion error in {filename}: {e}")
                continue
            bucket = get_buckets_for_position(idx, len(fens))[0]
            side = "white" if fen.split()[1] == "w" else "black"
            if winner == "draw":
                outcome = "draw"
            else:
                outcome = "win" if side == winner else "loss"
            yield is_val, (tensor, move, outcome, bucket, side)

class ShardWriter:
    def __init__(self, output_dir, prefix, shard_size=250_000):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_count = 0
        self.buffer = []

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        tensors, moves, values, buckets, sides = zip(*self.buffer)
        out_path = os.path.join(self.output_dir, f"{self.prefix}_{self.shard_count:03d}.pt")
        torch.save({
            "inputs": torch.stack(tensors),
            "policy": list(moves),
            "value": list(values),
            "buckets": list(buckets),
            "side_to_move": list(sides)
        }, out_path)
        print(f"💾 Saved {out_path} ({len(self.buffer)} samples)")
        self.shard_count += 1
        self.buffer.clear()

    def finalize(self):
        self.flush()

def merge_to_shards(fen_dir, move_dir, output_dir, max_games=None):
    os.makedirs(output_dir, exist_ok=True)
    game_files = sorted(f for f in os.listdir(fen_dir) if f.endswith(".txt"))
    if max_games:
        game_files = game_files[:max_games]

    random.shuffle(game_files)
    split = int(len(game_files) * (1 - VAL_FRACTION))
    train_files, val_files = game_files[:split], game_files[split:]
    val_files_set = set(val_files)

    print(f"Splitting games → Train: {len(train_files)} | Val: {len(val_files)}")

    train_writer = ShardWriter(output_dir, prefix="shard", shard_size=SHARD_SIZE)
    val_writer = ShardWriter(output_dir, prefix="val", shard_size=SHARD_SIZE)

    for is_val, sample in generate_samples(fen_dir, move_dir, val_files_set):
        (val_writer if is_val else train_writer).add(sample)

    train_writer.finalize()
    val_writer.finalize()

    print(f"\n✅ Done. Train shards: {train_writer.shard_count}, Val shards: {val_writer.shard_count}")

if __name__ == "__main__":
    merge_to_shards(fen_dir="fens/", move_dir="moves/", output_dir="shards/", max_games=None)
