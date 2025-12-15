import chess
import chess.pgn
import numpy as np
from typing import Iterator, Tuple, Optional
import struct

class PGNToNNUE:
    def __init__(self, pgn_path: str, min_rating: int = 1800, max_positions_per_game: int = 40):
        """
        Initialize PGN parser for NNUE training data generation.
        
        Args:
            pgn_path: Path to the PGN file
            min_rating: Minimum player rating to include games
            max_positions_per_game: Maximum positions to extract per game
        """
        self.pgn_path = pgn_path
        self.min_rating = min_rating
        self.max_positions_per_game = max_positions_per_game
    
    def parse_games(self) -> Iterator[chess.pgn.Game]:
        """Stream games from PGN file efficiently."""
        with open(self.pgn_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                yield game
    
    def filter_game(self, game: chess.pgn.Game) -> bool:
        """Check if game meets quality criteria."""
        headers = game.headers
        
        # Check if game has required headers
        if 'WhiteElo' not in headers or 'BlackElo' not in headers:
            return False
        
        try:
            white_elo = int(headers['WhiteElo'])
            black_elo = int(headers['BlackElo'])
        except ValueError:
            return False
        
        # Filter by minimum rating
        if white_elo < self.min_rating or black_elo < self.min_rating:
            return False
        
        # Filter out very short games (likely abandons)
        if len(list(game.mainline_moves())) < 20:
            return False
        
        return True
    
    def game_result_to_score(self, result: str, is_white_turn: bool) -> Optional[float]:
        """
        Convert game result to a score from current player's perspective.
        
        Returns:
            1.0 for win, 0.0 for loss, 0.5 for draw, None if unknown
        """
        if result == '1-0':  # White wins
            return 1.0 if is_white_turn else 0.0
        elif result == '0-1':  # Black wins
            return 0.0 if is_white_turn else 1.0
        elif result == '1/2-1/2':  # Draw
            return 0.5
        return None
    
    def board_to_features(self, board: chess.Board) -> np.ndarray:
        """
        Convert board to feature vector (simplified piece-square representation).
        
        This is a basic implementation. For production NNUE, you'd want
        HalfKP (King-Piece) features like Stockfish uses.
        
        Returns:
            768-dimensional feature vector (64 squares * 12 piece types)
        """
        features = np.zeros(768, dtype=np.float32)
        
        piece_to_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Calculate index: square * 12 + (piece_type - 1) * 2 + color
                piece_idx = piece_to_idx[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                idx = square * 12 + piece_idx + color_offset
                features[idx] = 1.0
        
        return features
    
    def extract_positions(self, game: chess.pgn.Game) -> list[Tuple[np.ndarray, float]]:
        """
        Extract training positions from a game.
        
        Returns:
            List of (features, score) tuples
        """
        result = game.headers.get('Result', '*')
        if result == '*':
            return []
        
        positions = []
        board = game.board()
        moves = list(game.mainline_moves())
        
        # Sample positions throughout the game
        step = max(1, len(moves) // self.max_positions_per_game)
        
        for i, move in enumerate(moves):
            if i % step == 0 and len(positions) < self.max_positions_per_game:
                # Skip opening positions (first 10 moves)
                if i >= 10:
                    score = self.game_result_to_score(result, board.turn)
                    if score is not None:
                        features = self.board_to_features(board)
                        positions.append((features, score))
            
            board.push(move)
        
        return positions
    
    def process_to_binary(self, output_path: str, max_games: Optional[int] = None):
        """
        Process PGN and save training data in binary format.
        
        Binary format: each record is 768 floats (features) + 1 float (score)
        """
        games_processed = 0
        positions_extracted = 0
        
        with open(output_path, 'wb') as out_file:
            for game in self.parse_games():
                if max_games and games_processed >= max_games:
                    break
                
                if self.filter_game(game):
                    positions = self.extract_positions(game)
                    
                    for features, score in positions:
                        # Write features (768 floats) and score (1 float)
                        out_file.write(features.tobytes())
                        out_file.write(struct.pack('f', score))
                        positions_extracted += 1
                    
                    games_processed += 1
                    
                    if games_processed % 1000 == 0:
                        print(f"Processed {games_processed} games, "
                              f"extracted {positions_extracted} positions")
        
        print(f"\nComplete! Processed {games_processed} games")
        print(f"Extracted {positions_extracted} training positions")
        print(f"Output saved to: {output_path}")
    
    def process_to_text(self, output_path: str, max_games: Optional[int] = None):
        """
        Process PGN and save training data in text format (FEN + score).
        Easier to inspect and debug.
        """
        games_processed = 0
        positions_extracted = 0
        
        with open(output_path, 'w') as out_file:
            for game in self.parse_games():
                if max_games and games_processed >= max_games:
                    break
                
                if self.filter_game(game):
                    result = game.headers.get('Result', '*')
                    board = game.board()
                    moves = list(game.mainline_moves())
                    step = max(1, len(moves) // self.max_positions_per_game)
                    
                    for i, move in enumerate(moves):
                        if i % step == 0 and i >= 10:
                            score = self.game_result_to_score(result, board.turn)
                            if score is not None:
                                fen = board.fen()
                                out_file.write(f"{fen}|{score}\n")
                                positions_extracted += 1
                        
                        board.push(move)
                    
                    games_processed += 1
                    
                    if games_processed % 1000 == 0:
                        print(f"Processed {games_processed} games, "
                              f"extracted {positions_extracted} positions")
        
        print(f"\nComplete! Processed {games_processed} games")
        print(f"Extracted {positions_extracted} training positions")
        print(f"Output saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize parser
    parser = PGNToNNUE(
        pgn_path='lichess_db_standard_rated_2025-01.pgn',
        min_rating=1800,
        max_positions_per_game=40
    )
    
    # Start with a small sample to test (e.g., 10,000 games)
    print("Processing PGN file...")
    print("Starting with 10,000 games for testing...")
    
    # Save as text format first (easier to inspect)
    parser.process_to_text('training_data.txt', max_games=10000)
    
    # For actual training, use binary format (much more efficient)
    # parser.process_to_binary('training_data.bin', max_games=10000)