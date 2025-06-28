import os
import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ChessPolicyValueNet(nn.Module):
    def __init__(self, input_planes, policy_size, num_blocks=12, channels=128):
        super(ChessPolicyValueNet, self).__init__()
        self.conv_in = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, policy_size)
        self.value_conv  = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(8 * 8 * 1, 256)
        self.value_fc2   = nn.Linear(256, 3)
        self.relu        = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        p = self.policy_conv(x)
        p = self.relu(self.policy_bn(p))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = self.value_conv(x)
        v = self.relu(self.value_bn(v))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        return p, v

class ChessDataset(Dataset):
    def __init__(self, shard_list, move_to_idx):
        self.inputs = torch.cat([shard["inputs"] for shard in shard_list])
        all_moves = []
        for shard in shard_list:
            all_moves.extend([move_to_idx[m] for m in shard["policy"]])
        self.policy_labels = torch.tensor(all_moves, dtype=torch.long)
        outcome_map = {"loss": 0, "draw": 1, "win": 2}
        all_values = []
        for shard in shard_list:
            all_values.extend([outcome_map[val] for val in shard["value"]])
        self.value_labels = torch.tensor(all_values, dtype=torch.long)
        self.buckets = []
        self.sides = []
        for shard in shard_list:
            self.buckets.extend(shard["buckets"])
            self.sides.extend(shard.get("side_to_move", ["unknown"] * len(shard["buckets"])))
        assert len(self.inputs) == len(self.policy_labels) == len(self.value_labels) == len(self.buckets) == len(self.sides)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx],
                self.policy_labels[idx],
                self.value_labels[idx],
                self.buckets[idx],
                self.sides[idx])

def load_move_index_map(path="../move_to_idx.json"):
    with open(path, "r") as f:
        move_to_idx = json.load(f)
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    return move_to_idx, idx_to_move

def top_k_accuracy(logits, targets, k=5):
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        correct = topk.eq(targets.unsqueeze(1))
        return correct.any(dim=1).float().mean().item()

def train_model(model, train_shard_paths, val_shards, move_to_idx, optimizer, scheduler,
                num_epochs=50, device="cpu", batch_size=256, start_epoch=1):
    policy_criterion = nn.CrossEntropyLoss(reduction='none')
    value_criterion  = nn.CrossEntropyLoss(reduction='none')
    bucket_weight = {
        "1-10": 1.0, "11-20": 1.2, "21-30": 1.3, "31-40": 1.3,
        "41-50": 1.3, "51-70": 1.2, "71+": 1.2, "mating": 1.0
    }
    val_dataset = ChessDataset(val_shards, move_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    best_val_policy_acc = 0.0

    for epoch in range(start_epoch, num_epochs+1):
        model.train()
        train_loss_sum = train_policy_correct = train_value_correct = train_total = 0
        side_correct = {"white": 0, "black": 0}
        side_total   = {"white": 0, "black": 0}

        for shard_path in train_shard_paths:
            shard = torch.load(shard_path, map_location='cpu')
            train_dataset = ChessDataset([shard], move_to_idx)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for batch_inputs, batch_policy, batch_value, batch_buckets, batch_sides in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_policy = batch_policy.to(device)
                batch_value  = batch_value.to(device)
                optimizer.zero_grad()
                policy_logits, value_logits = model(batch_inputs)
                policy_loss = policy_criterion(policy_logits, batch_policy)
                value_loss  = value_criterion(value_logits, batch_value)
                weights = torch.tensor([bucket_weight.get(b, 1.0) for b in batch_buckets], device=device)
                loss = ((policy_loss + value_loss) * weights).mean()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(batch_inputs)
                with torch.no_grad():
                    preds = policy_logits.argmax(dim=1)
                    train_policy_correct += (preds == batch_policy).sum().item()
                    val_preds = value_logits.argmax(dim=1)
                    train_value_correct += (val_preds == batch_value).sum().item()
                    for i, side in enumerate(batch_sides):
                        if side in side_correct:
                            if preds[i] == batch_policy[i]:
                                side_correct[side] += 1
                            side_total[side] += 1
                train_total += len(batch_inputs)
            del shard, train_dataset, train_loader
            torch.cuda.empty_cache()
            gc.collect()

        train_policy_acc = train_policy_correct / train_total
        train_value_acc = train_value_correct / train_total
        avg_loss = train_loss_sum / train_total
        white_acc = side_correct["white"] / side_total["white"] if side_total["white"] else 0
        black_acc = side_correct["black"] / side_total["black"] if side_total["black"] else 0

        print(f"Epoch {epoch}: Train Policy Acc = {train_policy_acc:.4f}, Train Value Acc = {train_value_acc:.4f}, Loss = {avg_loss:.4f}")
        print(f"            (White move acc: {white_acc:.4f}, Black move acc: {black_acc:.4f})")
        scheduler.step()

        model.eval()
        val_policy_correct = val_value_correct = val_policy_top5_correct = val_total = 0
        side_correct_val = {"white": 0, "black": 0}
        side_total_val = {"white": 0, "black": 0}

        with torch.no_grad():
            for batch_inputs, batch_policy, batch_value, batch_buckets, batch_sides in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_policy = batch_policy.to(device)
                batch_value  = batch_value.to(device)
                policy_logits, value_logits = model(batch_inputs)
                preds = policy_logits.argmax(dim=1)
                val_policy_correct += (preds == batch_policy).sum().item()
                val_policy_top5_correct += policy_logits.topk(5, dim=1).indices.eq(batch_policy.unsqueeze(1)).any(dim=1).sum().item()
                val_preds = value_logits.argmax(dim=1)
                val_value_correct += (val_preds == batch_value).sum().item()
                for i, side in enumerate(batch_sides):
                    if side in side_correct_val:
                        if preds[i] == batch_policy[i]:
                            side_correct_val[side] += 1
                        side_total_val[side] += 1
                val_total += len(batch_inputs)

        val_policy_acc = val_policy_correct / val_total
        val_policy_top5 = val_policy_top5_correct / val_total
        val_value_acc = val_value_correct / val_total
        white_val_acc = side_correct_val["white"] / side_total_val["white"] if side_total_val["white"] else 0
        black_val_acc = side_correct_val["black"] / side_total_val["black"] if side_total_val["black"] else 0

        print(f"         Validation: Policy Top-1 = {val_policy_acc:.4f}, Policy Top-5 = {val_policy_top5:.4f}, Value = {val_value_acc:.4f}")
        print(f"            (White move acc: {white_val_acc:.4f}, Black move acc: {black_val_acc:.4f})")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "move_to_idx": move_to_idx,
            "val_policy_acc": val_policy_acc,
            "val_value_acc": val_value_acc
        }
        torch.save(checkpoint, "checkpoint_last.pt")
        if val_policy_acc > best_val_policy_acc:
            best_val_policy_acc = val_policy_acc
            torch.save(checkpoint, "checkpoint_best.pt")
            print(f"📂 Saved new best model (epoch {epoch}, policy acc {val_policy_acc:.4f})")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards_dir = "shards/"
    move_index_path = "move_to_idx.json"

    if not os.path.exists(move_index_path):
        raise FileNotFoundError("move_to_idx.json not found. Please run move2index.py to generate the move index mapping.")
    move_to_idx, idx_to_move = load_move_index_map(move_index_path)
    policy_size = len(move_to_idx)

    model = ChessPolicyValueNet(input_planes=17, policy_size=policy_size, num_blocks=12, channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min((epoch+1)/5, 1.0))
    start_epoch = 1

    if os.path.exists("checkpoint_last.pt"):
        checkpoint = torch.load("checkpoint_last.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"\n🔁 Resuming from epoch {checkpoint['epoch']}...\n")
    else:
        print(f"\n🌐 Starting fresh training...\n")

    print(f"Initialized model with {len(model.res_blocks)} residual blocks and {model.policy_fc.out_features} policy outputs.")

    train_shard_paths = [os.path.join(shards_dir, fname) for fname in sorted(os.listdir(shards_dir)) if fname.startswith("shard_") and fname.endswith(".pt")]
    val_shards = [torch.load(os.path.join(shards_dir, fname), map_location='cpu') for fname in sorted(os.listdir(shards_dir)) if fname.startswith("val_") and fname.endswith(".pt")]

    if not train_shard_paths or not val_shards:
        raise RuntimeError("No training/validation shards found. Make sure data_preparation ran correctly.")

    print(f"Training on {len(train_shard_paths)} shards, validating on {sum(len(s['policy']) for s in val_shards)} samples.")

    train_model(model, train_shard_paths, val_shards, move_to_idx,
                optimizer=optimizer, scheduler=scheduler,
                num_epochs=100, device=device, batch_size=256, start_epoch=start_epoch)

