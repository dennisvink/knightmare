# ♞ Knightmare

**Knightmare** is a minimal chess engine combining **deep learning** and **Monte Carlo Tree Search (MCTS)** to play full games of chess through a browser interface.

This project features:
- A **PyTorch-based neural network** that evaluates positions and suggests policies.
- An **MCTS** search strategy guided by network predictions.
- A lightweight **Flask** backend serving moves and game state to the browser.
- Optional deployment via **Docker**.

---

## Demo of Knightmare

https://chessbot-uo7slb3u7a-uc.a.run.app/

## How It Works

Knightmare uses a neural network to evaluate board positions:

1. **Board Representation**: A custom tensor encoding from FEN using 17 channels (12 piece planes, 4 castling rights, 1 en-passant).
2. **Neural Network**: A residual CNN outputs:
   - A **policy** over legal moves.
   - A **value** prediction (win, draw, loss).
3. **MCTS**: A guided search tree is built using:
   - The **value head** to backpropagate outcomes.
   - The **policy head** to inform expansion priors.
4. The move with the highest visit count is selected after N simulations.

---

## Getting Started Locally

To run Knightmare locally without Docker:

### 1. Clone the repository
```bash
git clone https://github.com/dennisvink/knightmare.git
cd knightmare
```

### 2. Set up a virtual environment
```bash
python -m venv .venv/
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure model and move index files are present
Make sure these files exist in the project root:
- `model.pt` – pre-trained model weights
- `move_to_idx.json` – mapping from move UCI strings to network indices

### 5. Run the server
```bash
python app.py
```

Then open your browser to [http://localhost:8080](http://localhost:8080)

---

## Running with Docker

### 1. Build the Docker image
```bash
docker build -t knightmare .
```

### 2. Run the container
```bash
docker run -p 8080:8080 knightmare
```

Visit [http://localhost:8080](http://localhost:8080) to play!

---

## 📁 Project Structure

```
├── app.py                # Flask application and game logic
├── model.pt              # Pre-trained PyTorch model
├── move_to_idx.json      # Mapping of UCI moves to model indices
├── templates/
│   └── index.html        # Web UI
├── static/
│   └── favicon.ico       # Favicon
├── requirements.txt      # Python dependencies
└── Dockerfile            # Docker build configuration
```

---

## Requirements

- Python 3.8+
- PyTorch
- Flask
- NumPy
- python-chess

(Dependencies are installed via `requirements.txt`)

---

Feel free to contribute, fork, or build upon Knightmare!
