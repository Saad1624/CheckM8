"""
Alpha-Beta Chess Engine with NNUE Evaluation
=============================================

This module implements a chess engine using:
- Alpha-Beta pruning for efficient search
- NNUE model for position evaluation
- Move ordering for better performance
- Iterative deepening for time management
- Quiescence search for tactical stability
"""

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from typing import Optional, Tuple, List
import time
from dataclasses import dataclass


# ============================================================================
# NNUE Model Architecture
# ============================================================================

class ImprovedNNUE(nn.Module):
    """NNUE architecture - same as training"""
    
    class ResidualBlock(nn.Module):
        def __init__(self, in_features, out_features, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(in_features, out_features)
            self.bn1 = nn.BatchNorm1d(out_features)
            self.fc2 = nn.Linear(out_features, out_features)
            self.bn2 = nn.BatchNorm1d(out_features)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.skip = nn.Linear(in_features, out_features) if in_features != out_features else None
        
        def forward(self, x):
            identity = x
            out = self.fc1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.bn2(out)
            if self.skip is not None:
                identity = self.skip(identity)
            out += identity
            return self.relu(out)
    
    def __init__(self, use_advanced_features=True):
        super().__init__()
        self.use_advanced_features = use_advanced_features
        input_size = 768 + (16 if use_advanced_features else 0)
        
        self.input_layer = nn.Linear(input_size, 512)
        self.input_bn = nn.BatchNorm1d(512)
        self.res_block1 = self.ResidualBlock(512, 384, dropout=0.15)
        self.res_block2 = self.ResidualBlock(384, 256, dropout=0.15)
        self.res_block3 = self.ResidualBlock(256, 128, dropout=0.1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)


# ============================================================================
# Feature Extraction
# ============================================================================

def fen_to_features(fen: str) -> np.ndarray:
    """Convert FEN to 768-dimensional piece-square features."""
    board = chess.Board(fen)
    features = np.zeros(768, dtype=np.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_idx = piece_to_idx[piece.piece_type]
            if not piece.color:  # Black
                piece_idx += 6
            feature_idx = piece_idx * 64 + square
            features[feature_idx] = 1.0
    
    return features


def extract_advanced_features(fen: str) -> np.ndarray:
    """Extract 16 strategic features."""
    board = chess.Board(fen)
    features = []
    
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9
    }
    
    # Material (2)
    for color in [chess.WHITE, chess.BLACK]:
        material = sum(len(board.pieces(pt, color)) * val for pt, val in piece_values.items())
        features.append(material / 39.0)
    
    # Development (2)
    for color in [chess.WHITE, chess.BLACK]:
        back_rank = 0 if color == chess.WHITE else 7
        developed = sum(1 for pt in [chess.KNIGHT, chess.BISHOP] 
                       for sq in board.pieces(pt, color) 
                       if chess.square_rank(sq) != back_rank)
        features.append(developed / 4.0)
    
    # King safety (2)
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
            pawns_near = sum(1 for df in [-1, 0, 1] for dr in [-1, 0, 1]
                           if 0 <= kf+df < 8 and 0 <= kr+dr < 8
                           and (p := board.piece_at(chess.square(kf+df, kr+dr)))
                           and p.piece_type == chess.PAWN and p.color == color)
            features.append(pawns_near / 8.0)
        else:
            features.append(0.0)
    
    # Castling rights (4)
    features.extend([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK))
    ])
    
    # Center control (2)
    center = [chess.E4, chess.E5, chess.D4, chess.D5]
    for color in [chess.WHITE, chess.BLACK]:
        control = sum(1 for sq in center if board.is_attacked_by(color, sq))
        features.append(control / 4.0)
    
    # Mobility (1)
    features.append(min(board.legal_moves.count() / 50.0, 1.0))
    
    # Turn (1)
    features.append(float(board.turn))
    
    # In check (1)
    features.append(float(board.is_check()))
    
    # Game phase (1)
    total_mat = sum(len(board.pieces(pt, c)) * val 
                   for c in [chess.WHITE, chess.BLACK]
                   for pt, val in piece_values.items())
    features.append(min(total_mat / 30.0, 1.0))
    
    return np.array(features, dtype=np.float32)


def board_to_features(board: chess.Board, use_advanced_features: bool = True) -> np.ndarray:
    """Convert board to full feature vector."""
    basic = fen_to_features(board.fen())
    if use_advanced_features:
        advanced = extract_advanced_features(board.fen())
        return np.concatenate([basic, advanced])
    return basic


# ============================================================================
# NNUE Evaluator
# ============================================================================

class NNUEEvaluator:
    """Wrapper for NNUE model evaluation."""
    
    def __init__(self, model_path: str = 'best_nnue_model.pth', 
                 use_onnx: bool = False, use_advanced_features: bool = True):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model (.pth or .onnx)
            use_onnx: Use ONNX runtime (faster inference)
            use_advanced_features: Use advanced features (must match training)
        """
        self.use_advanced_features = use_advanced_features
        self.use_onnx = use_onnx
        
        if use_onnx:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            print(f"✓ Loaded ONNX model from {model_path}")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = ImprovedNNUE(use_advanced_features=use_advanced_features)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded PyTorch model from {model_path} (device: {self.device})")
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position from current player's perspective.
        
        Returns:
            Score in centipawns (positive = good for side to move)
        """
        features = board_to_features(board, self.use_advanced_features)
        
        if self.use_onnx:
            # ONNX inference
            features_tensor = features.reshape(1, -1).astype(np.float32)
            score = self.session.run(None, {self.input_name: features_tensor})[0][0][0]
        else:
            # PyTorch inference
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                score = self.model(features_tensor).item()
        
        # Convert to centipawns from current player's perspective
        # Score is for white, so flip if black to move
        cp = self.score_to_centipawns(score)
        if not board.turn:  # Black to move
            cp = -cp
        
        return cp
    
    @staticmethod
    def score_to_centipawns(score: float, k: float = 1/400) -> float:
        """Convert NNUE score (0-1) to centipawns."""
        if score >= 0.999:
            return 10000
        if score <= 0.001:
            return -10000
        cp = -np.log10((1/score) - 1) / k * 100
        return cp


# ============================================================================
# Move Ordering
# ============================================================================

class MoveOrderer:
    """Order moves for better alpha-beta pruning."""
    
    # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    @classmethod
    def order_moves(cls, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Order moves by priority:
        1. Captures (MVV-LVA)
        2. Promotions
        3. Checks
        4. Other moves
        """
        def move_score(move):
            score = 0
            
            # Captures (most valuable victim first)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += cls.PIECE_VALUES[victim.piece_type] * 10
                    score -= cls.PIECE_VALUES[attacker.piece_type]
            
            # Promotions
            if move.promotion:
                score += cls.PIECE_VALUES[move.promotion]
            
            # Checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            # Castle
            if board.is_castling(move):
                score += 100
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)


# ============================================================================
# Alpha-Beta Search Engine
# ============================================================================

@dataclass
class SearchStats:
    """Statistics for search."""
    nodes_searched: int = 0
    time_elapsed: float = 0.0
    depth_reached: int = 0
    best_move: Optional[chess.Move] = None
    best_score: float = 0.0


class ChessEngine:
    """Chess engine with alpha-beta search and NNUE evaluation."""
    
    def __init__(self, evaluator: NNUEEvaluator, max_depth: int = 5, 
                 use_quiescence: bool = True):
        """
        Initialize engine.
        
        Args:
            evaluator: NNUE evaluator
            max_depth: Maximum search depth
            use_quiescence: Use quiescence search for tactical positions
        """
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.use_quiescence = use_quiescence
        self.stats = SearchStats()
    
    def search(self, board: chess.Board, depth: Optional[int] = None, 
               time_limit: Optional[float] = None) -> Tuple[chess.Move, float]:
        """
        Find best move using iterative deepening alpha-beta search.
        
        Args:
            board: Current position
            depth: Search depth (None = use max_depth)
            time_limit: Time limit in seconds (None = no limit)
        
        Returns:
            (best_move, evaluation)
        """
        if depth is None:
            depth = self.max_depth
        
        self.stats = SearchStats()
        start_time = time.time()
        
        best_move = None
        best_score = float('-inf')
        
        # Iterative deepening
        for current_depth in range(1, depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            score, move = self._alpha_beta_root(board, current_depth, start_time, time_limit)
            
            if move is not None:
                best_move = move
                best_score = score
                self.stats.depth_reached = current_depth
            
            # Time check
            if time_limit and (time.time() - start_time) > time_limit:
                break
        
        self.stats.time_elapsed = time.time() - start_time
        self.stats.best_move = best_move
        self.stats.best_score = best_score
        
        return best_move, best_score
    
    def _alpha_beta_root(self, board: chess.Board, depth: int, 
                         start_time: float, time_limit: Optional[float]) -> Tuple[float, Optional[chess.Move]]:
        """Root node of alpha-beta search."""
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        best_score = float('-inf')
        
        moves = list(board.legal_moves)
        moves = MoveOrderer.order_moves(board, moves)
        
        for move in moves:
            # Time check
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, start_time, time_limit)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_score, best_move
    
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float,
                    start_time: float, time_limit: Optional[float]) -> float:
        """
        Alpha-beta pruning search.
        
        Args:
            board: Current position
            depth: Remaining depth
            alpha: Alpha value (best score for maximizer)
            beta: Beta value (best score for minimizer)
            start_time: Search start time
            time_limit: Time limit
        
        Returns:
            Evaluation score
        """
        self.stats.nodes_searched += 1
        
        # Time check
        if time_limit and (time.time() - start_time) > time_limit:
            return 0.0
        
        # Terminal conditions
        if board.is_checkmate():
            return -10000 - depth  # Prefer faster mates
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0
        
        # Depth limit
        if depth <= 0:
            if self.use_quiescence:
                return self._quiescence(board, alpha, beta, start_time, time_limit)
            else:
                return self.evaluator.evaluate(board)
        
        # Generate and order moves
        moves = list(board.legal_moves)
        moves = MoveOrderer.order_moves(board, moves)
        
        # Alpha-beta search
        for move in moves:
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, start_time, time_limit)
            board.pop()
            
            if score >= beta:
                return beta  # Beta cutoff
            
            alpha = max(alpha, score)
        
        return alpha
    
    def _quiescence(self, board: chess.Board, alpha: float, beta: float,
                    start_time: float, time_limit: Optional[float], depth: int = 0) -> float:
        """
        Quiescence search - only consider tactical moves (captures, checks).
        Prevents horizon effect.
        """
        self.stats.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Quiescence depth limit
        if depth >= 10:
            return stand_pat
        
        # Only consider captures and checks
        moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        moves = MoveOrderer.order_moves(board, moves)
        
        for move in moves:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, start_time, time_limit, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            if score > alpha:
                alpha = score
        
        return alpha
    
    def print_stats(self):
        """Print search statistics."""
        print(f"\n{'='*60}")
        print(f"SEARCH STATISTICS")
        print(f"{'='*60}")
        print(f"Best move:       {self.stats.best_move}")
        print(f"Evaluation:      {self.stats.best_score:+.2f} centipawns")
        print(f"Depth reached:   {self.stats.depth_reached}")
        print(f"Nodes searched:  {self.stats.nodes_searched:,}")
        print(f"Time elapsed:    {self.stats.time_elapsed:.2f}s")
        if self.stats.time_elapsed > 0:
            nps = self.stats.nodes_searched / self.stats.time_elapsed
            print(f"Nodes/second:    {nps:,.0f}")
        print(f"{'='*60}\n")


# ============================================================================
# Example Usage and Interactive Play
# ============================================================================

def play_game_interactive():
    """Interactive game against the engine."""
    print("\n" + "="*60)
    print("CHESS ENGINE - INTERACTIVE PLAY")
    print("="*60)
    print("\nInitializing engine...")
    
    # Initialize evaluator (use ONNX for faster inference)
    try:
        evaluator = NNUEEvaluator('nnue_model.onnx', use_onnx=True, use_advanced_features=True)
    except:
        print("ONNX model not found, using PyTorch model...")
        evaluator = NNUEEvaluator('best_nnue_model.pth', use_onnx=False, use_advanced_features=True)
    
    # Initialize engine
    engine = ChessEngine(evaluator, max_depth=5, use_quiescence=True)
    
    board = chess.Board()
    
    print("\nEngine ready! You are White.")
    print("Enter moves in UCI format (e.g., 'e2e4')")
    print("Type 'quit' to exit, 'fen' to see FEN, 'eval' for evaluation\n")
    
    while not board.is_game_over():
        print(board)
        print(f"\nFEN: {board.fen()}")
        
        if board.turn:  # White (human)
            print("\nYour move (White): ", end="")
            move_str = input().strip().lower()
            
            if move_str == 'quit':
                break
            elif move_str == 'fen':
                print(f"FEN: {board.fen()}")
                continue
            elif move_str == 'eval':
                eval_score = evaluator.evaluate(board)
                print(f"Evaluation: {eval_score:+.2f} centipawns")
                continue
            
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
                    continue
            except:
                print("Invalid move format! Use UCI (e.g., 'e2e4')")
                continue
        
        else:  # Black (engine)
            print("\nEngine thinking...")
            best_move, score = engine.search(board, depth=5, time_limit=10.0)
            
            if best_move:
                print(f"Engine plays: {best_move} (eval: {score:+.2f} cp)")
                engine.print_stats()
                board.push(best_move)
            else:
                print("Engine couldn't find a move!")
                break
    
    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    print(f"Result: {board.result()}")
    if board.is_checkmate():
        winner = "White" if not board.turn else "Black"
        print(f"{winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Draw by stalemate")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    print(board)


def analyze_position(fen: str, depth: int = 6):
    """Analyze a specific position."""
    print("\n" + "="*60)
    print("POSITION ANALYSIS")
    print("="*60)
    
    board = chess.Board(fen)
    print(f"\n{board}\n")
    print(f"FEN: {fen}\n")
    
    # Initialize
    try:
        evaluator = NNUEEvaluator('nnue_model.onnx', use_onnx=True, use_advanced_features=True)
    except:
        evaluator = NNUEEvaluator('best_nnue_model.pth', use_onnx=False, use_advanced_features=True)
    
    engine = ChessEngine(evaluator, max_depth=depth, use_quiescence=True)
    
    # Static evaluation
    static_eval = evaluator.evaluate(board)
    print(f"Static evaluation: {static_eval:+.2f} centipawns\n")
    
    # Search for best move
    print(f"Searching to depth {depth}...")
    best_move, score = engine.search(board, depth=depth)
    
    print(f"\nBest move: {best_move}")
    print(f"Evaluation: {score:+.2f} centipawns")
    engine.print_stats()
    
    # Show top 3 moves
    print("\nTop moves:")
    moves = list(board.legal_moves)
    move_scores = []
    for move in moves[:10]:  # Check top 10 by move ordering
        board.push(move)
        eval_score = -evaluator.evaluate(board)
        board.pop()
        move_scores.append((move, eval_score))
    
    move_scores.sort(key=lambda x: x[1], reverse=True)
    for i, (move, score) in enumerate(move_scores[:3], 1):
        print(f"  {i}. {move}: {score:+.2f} cp")


if __name__ == "__main__":
    print("Chess Engine with NNUE and Alpha-Beta Search")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Play against engine (interactive)")
    print("2. Analyze a position")
    print("3. Quick test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        play_game_interactive()
    elif choice == "2":
        fen = input("Enter FEN (or press Enter for starting position): ").strip()
        if not fen:
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        depth = int(input("Enter search depth (3-8 recommended): ") or "5")
        analyze_position(fen, depth)
    else:
        # Quick test
        print("\nRunning quick test...")
        analyze_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=4)