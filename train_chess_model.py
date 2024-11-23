import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model using PyTorch
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(773, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

model = ChessNet()

# Function to convert board state to input features
def board_to_input(board):
    # One-hot encoding of pieces on the board plus additional features
    piece_map = board.piece_map()
    x = np.zeros(773)  # 64 squares * 12 piece types + 5 additional features
    for square, piece in piece_map.items():
        idx = square * 12 + (piece.piece_type - 1)
        if piece.color == chess.BLACK:
            idx += 6
        x[idx] = 1
    # Additional features
    x[768] = int(board.turn)  # 1 for White, 0 for Black
    x[769] = int(board.has_kingside_castling_rights(chess.WHITE))
    x[770] = int(board.has_queenside_castling_rights(chess.WHITE))
    x[771] = int(board.has_kingside_castling_rights(chess.BLACK))
    x[772] = int(board.has_queenside_castling_rights(chess.BLACK))
    return x

# Simple evaluation function based on material count
def simple_evaluation(board):
    # Piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King's value is set to 0 for evaluation purposes
    }
    material = 0
    for piece in board.piece_map().values():
        value = piece_values[piece.piece_type]
        if piece.color == chess.WHITE:
            material += value
        else:
            material -= value
    return material / 39  # Normalize to [-1, 1]

# Generate training data
def generate_training_data(num_samples=10000):
    X = []
    y = []
    for _ in range(num_samples):
        board = chess.Board()
        while not board.is_game_over():
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
            X.append(board_to_input(board))
            y.append(simple_evaluation(board))
            if len(X) >= num_samples:
                break
        if len(X) >= num_samples:
            break
    return np.array(X), np.array(y)

# Training the model
def train_model():
    print("Generating training data...")
    X_train, y_train = generate_training_data(num_samples=10000)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Training settings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000
    batch_size = 32

    # Training loop
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        average_loss = epoch_loss / len(loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.6f}')
    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), 'chess_model.pth')
    print("Model saved to 'chess_model.pth'.")

if __name__ == '__main__':
    train_model()
