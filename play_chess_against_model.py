import pygame
import sys
import chess
import numpy as np
import torch
import torch.nn as nn

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
model.load_state_dict(torch.load('chess_model.pth'))
model.eval()

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

# Evaluation function using the neural network
def evaluate(board):
    x = board_to_input(board)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        score = model(x_tensor)
    return score.item()

# Simple minimax search with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Function to choose the best move
def choose_move(board, depth=2):
    best_move = None
    best_value = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, alpha, beta, False)
        board.pop()
        if board_value > best_value:
            best_value = board_value
            best_move = move
        alpha = max(alpha, board_value)
        if beta <= alpha:
            break
    return best_move

# Mapping of pieces to Unicode symbols
PIECE_UNICODE = {
    'P': '\u2659',
    'N': '\u2658',
    'B': '\u2657',
    'R': '\u2656',
    'Q': '\u2655',
    'K': '\u2654',
    'p': '\u265F',
    'n': '\u265E',
    'b': '\u265D',
    'r': '\u265C',
    'q': '\u265B',
    'k': '\u265A',
}

def main():
    # Initialize Pygame
    pygame.init()

    # Set up the display
    WIDTH, HEIGHT = 512, 512
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess Engine')

    # Load fonts
    def load_font(size):
        try:
            # Try to use a system font that supports chess symbols
            font = pygame.font.SysFont('DejaVu Sans', size)
        except:
            # Fallback to default font
            font = pygame.font.Font(None, size)
        return font

    FONT_SIZE = WIDTH // 8
    FONT = load_font(FONT_SIZE)

    clock = pygame.time.Clock()
    board = chess.Board()
    selected_square = None
    running = True
    player_turn = True  # Player plays as White

    # Draw the board
    def draw_board(screen, board, selected_square=None):
        colors = [pygame.Color(235, 235, 208), pygame.Color(119, 149, 86)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                rect = pygame.Rect(col * WIDTH // 8, row * HEIGHT // 8, WIDTH // 8, HEIGHT // 8)
                pygame.draw.rect(screen, color, rect)
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                if piece:
                    piece_symbol = piece.symbol()
                    text = PIECE_UNICODE[piece_symbol]
                    text_surface = FONT.render(text, True, pygame.Color('black'))
                    text_rect = text_surface.get_rect(center=rect.center)
                    screen.blit(text_surface, text_rect)
                # Highlight selected square
                if selected_square == square:
                    pygame.draw.rect(screen, pygame.Color('yellow'), rect, 3)

    def get_square_under_mouse():
        mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
        x, y = [int(v // (WIDTH / 8)) for v in mouse_pos]
        if x >= 0 and x < 8 and y >= 0 and y < 8:
            return chess.square(x, 7 - y)
        else:
            return None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if player_turn and not board.is_game_over():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    square = get_square_under_mouse()
                    if square is not None:
                        piece = board.piece_at(square)
                        if piece and piece.color == chess.WHITE:
                            selected_square = square

                elif event.type == pygame.MOUSEBUTTONUP:
                    if selected_square is not None:
                        square = get_square_under_mouse()
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                            # Check if the game is over after player's move
                            if board.is_game_over():
                                running = False
                            else:
                                player_turn = False
                        else:
                            # Try promotion moves
                            promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                            for p in promotion_pieces:
                                move = chess.Move(selected_square, square, promotion=p)
                                if move in board.legal_moves:
                                    board.push(move)
                                    selected_square = None
                                    # Check if the game is over after player's move
                                    if board.is_game_over():
                                        running = False
                                    else:
                                        player_turn = False
                                    break
                            else:
                                selected_square = None

        # Engine's turn
        if not player_turn and not board.is_game_over():
            print("Engine is thinking...")
            engine_move = choose_move(board, depth=2)
            if engine_move is not None:
                board.push(engine_move)
                print(f"Engine's move: {engine_move}")
                # Check if the game is over after engine's move
                if board.is_game_over():
                    running = False
                else:
                    player_turn = True
            else:
                print("Engine resigns.")
                running = False  # End the game if the engine cannot move

        # Draw the game state
        draw_board(WINDOW, board, selected_square)
        pygame.display.flip()
        clock.tick(60)

    # Game over
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            print("Engine wins!")
        else:
            print("You win!")
    elif board.is_stalemate():
        print("It's a stalemate!")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw due to the seventy-five moves rule.")
    elif board.is_fivefold_repetition():
        print("Draw due to fivefold repetition.")

    # Keep the window open for a short time to see the final position
    pygame.time.wait(5000)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
