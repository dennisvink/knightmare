# Knightmare

**Knightmare** is a minimal chess engine written by Dennis Vink (drvink.com) combining **deep learning** and **Monte Carlo Tree Search (MCTS)** to play full games of chess through a browser interface.

This project features:
- A **PyTorch-based neural network** that evaluates board positions.
- A **Monte Carlo Tree Search (MCTS)** planner for strong move selection.
- A **Flask web interface** to interact with the engine.
- A complete **training pipeline** to generate and learn from chess data.
- Optional **Docker** containerization for easy deployment.

---

## Demo

- https://huggingface.co/spaces/dennisvink/knightmare
- https://chessbot-uo7slb3u7a-uc.a.run.app/

---

## How It Works

1. **Board Representation**: FEN strings are encoded into 17-channel tensors (12 piece types, 4 castling planes, 1 en-passant).
2. **Neural Network**: A deep residual convolutional network estimates:
   - **Policy**: probability distribution over legal moves.
   - **Value**: predicted outcome (win/draw/loss).
3. **MCTS**: The network guides a Monte Carlo Tree Search to simulate positions and return the best move.
4. **Flask API**: Communicates between frontend and engine to serve moves interactively.

---

## Getting Started Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/knightmare.git
cd knightmare
```

### 2. Set up a virtual environment
```bash
python -m venv .venv/
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure model and index exist
Ensure the following files exist in the project root:
- `model.pt` – trained model weights
- `move_to_idx.json` – dictionary mapping UCI moves to index positions

### 5. Run the server
```bash
python app.py
```

Navigate to [http://localhost:7860](http://localhost:7860) to play!

---

## Docker

### Build the image
```bash
docker build -t knightmare .
```

### Run the container
```bash
docker run -p 7860:7860 knightmare
```

Visit [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🎓 Training Pipeline

You can train Knightmare from scratch using your own labeled games.

### Step 1: Prepare Your Data

Under the `pipeline/` directory:

- `fens/`: Each file contains one game with a list of FENs, one per line. First line must be `# Result 1-0`, `0-1`, or `1/2-1/2`.
- `moves/`: Each file has the corresponding best move per position (UCI format), one per line. Use `none` for terminal positions.

**Example:**
```
fens/game000000000.txt
moves/game000000000.moves.txt
```

### Step 2: Preprocess into shards
```bash
cd pipeline/
python data_preparation.py
```

This creates `shards/` with training and validation `.pt` files.

### Step 3: Train the model
```bash
python train.py
```

- Will resume from `checkpoint_last.pt` if available.
- Saves best model as `checkpoint_best.pt`.

---

## 📁 Project Structure

```
├── app.py
├── model.pt
├── move_to_idx.json
├── requirements.txt
├── Dockerfile
├── templates/
│   └── index.html
├── static/
│   └── favicon.ico
└── pipeline/
    ├── fens/
    ├── moves/
    ├── shards/
    ├── data_preparation.py
    └── train.py
```

---

## Requirements

- Python 3.8+
- PyTorch
- Flask
- python-chess
- NumPy
- tqdm

Installable via `pip install -r requirements.txt`.

---

## Contributing

Knightmare is an educational chess engine meant to grow. Feel free to fork, modify, and experiment.
