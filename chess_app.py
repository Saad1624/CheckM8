"""
NNUE Chess Engine - Streamlit Web Interface
============================================

A beautiful, interactive web interface for your NNUE chess engine.
100% Pure Python!

Usage:
    streamlit run chess_app.py

Then open browser to: http://localhost:8501
"""

import streamlit as st
import chess
import chess.svg
import chess.pgn
from io import StringIO
import base64
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import time

# Import your engine (make sure these files are in same directory)
try:
    from alpha_beta import NNUEEvaluator, ChessEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    st.error("⚠️ alpha_beta_engine.py not found! Please ensure it's in the same directory.")


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="NNUE Chess Engine",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    
    if 'move_history' not in st.session_state:
        st.session_state.move_history = []
    
    if 'eval_history' not in st.session_state:
        st.session_state.eval_history = []
    
    if 'engine' not in st.session_state and ENGINE_AVAILABLE:
        try:
            # Try ONNX first (faster)
            evaluator = NNUEEvaluator('models/nnue_model.onnx', use_onnx=True, use_advanced_features=True)
            st.session_state.evaluator = evaluator
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.session_state.evaluator = None
    
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    
    if 'player_color' not in st.session_state:
        st.session_state.player_color = chess.WHITE
    
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False


initialize_session_state()


# ============================================================================
# Helper Functions
# ============================================================================

def board_to_svg(board, last_move=None, check_square=None):
    """Convert board to SVG with highlighting."""
    squares = None
    
    # Highlight last move
    if last_move:
        squares = chess.SquareSet([last_move.from_square, last_move.to_square])
    
    # Highlight check
    if check_square:
        if squares:
            squares.add(check_square)
        else:
            squares = chess.SquareSet([check_square])
    
    svg = chess.svg.board(
        board,
        squares=squares,
        size=400,
        orientation=st.session_state.player_color
    )
    return svg


def svg_to_html(svg_string):
    """Convert SVG string to HTML for display."""
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
    return html


def format_move(move, board):
    """Format move in SAN notation."""
    return board.san(move)


def get_game_status(board):
    """Get current game status."""
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        return f"🏆 Checkmate! {winner} wins!"
    elif board.is_stalemate():
        return "🤝 Draw by stalemate"
    elif board.is_insufficient_material():
        return "🤝 Draw by insufficient material"
    elif board.can_claim_fifty_moves():
        return "🤝 Draw by 50-move rule"
    elif board.can_claim_threefold_repetition():
        return "🤝 Draw by repetition"
    elif board.is_check():
        return "⚠️ Check!"
    else:
        return "🎮 Game in progress"


def make_engine_move(depth, time_limit=None):
    """Make engine move with progress indication."""
    if st.session_state.evaluator is None:
        st.error("Engine not available!")
        return
    
    engine = ChessEngine(
        st.session_state.evaluator, 
        max_depth=depth,
        use_quiescence=True
    )
    
    # Search for best move
    start_time = time.time()
    best_move, score = engine.search(
        st.session_state.board, 
        depth=depth,
        time_limit=time_limit
    )
    elapsed_time = time.time() - start_time
    
    if best_move:
        # Record move
        san_move = format_move(best_move, st.session_state.board)
        st.session_state.board.push(best_move)
        st.session_state.move_history.append({
            'move': san_move,
            'uci': best_move.uci(),
            'evaluation': score,
            'depth': depth,
            'nodes': engine.stats.nodes_searched,
            'time': elapsed_time
        })
        st.session_state.eval_history.append(score)
        
        return best_move, score, engine.stats
    
    return None, None, None


# ============================================================================
# Sidebar - Game Controls
# ============================================================================

with st.sidebar:
    st.title("♟️ NNUE Chess Engine")
    st.markdown("---")
    
    # New Game Section
    st.subheader("🎮 New Game")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚪ Play as White", use_container_width=True):
            st.session_state.board = chess.Board()
            st.session_state.move_history = []
            st.session_state.eval_history = []
            st.session_state.player_color = chess.WHITE
            st.session_state.game_started = True
            st.rerun()
    
    with col2:
        if st.button("⚫ Play as Black", use_container_width=True):
            st.session_state.board = chess.Board()
            st.session_state.move_history = []
            st.session_state.eval_history = []
            st.session_state.player_color = chess.BLACK
            st.session_state.game_started = True
            # Engine makes first move
            st.session_state.thinking = True
            st.rerun()
    
    if st.button("🔄 Reset Board", use_container_width=True):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.eval_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Engine Settings
    st.subheader("⚙️ Engine Settings")
    
    engine_depth = st.slider(
        "Search Depth",
        min_value=1,
        max_value=8,
        value=5,
        help="Higher = stronger but slower"
    )
    
    time_limit = st.slider(
        "Time Limit (seconds)",
        min_value=1,
        max_value=60,
        value=10,
        help="Maximum time per move"
    )
    
    show_analysis = st.checkbox("Show Analysis", value=True)
    
    st.markdown("---")
    
    # Game Actions
    st.subheader("🎯 Game Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Undo Move", use_container_width=True):
            if len(st.session_state.board.move_stack) >= 2:
                # Undo player and engine moves
                st.session_state.board.pop()
                st.session_state.board.pop()
                if len(st.session_state.move_history) >= 2:
                    st.session_state.move_history.pop()
                    st.session_state.move_history.pop()
                if len(st.session_state.eval_history) >= 2:
                    st.session_state.eval_history.pop()
                    st.session_state.eval_history.pop()
                st.rerun()
    
    with col2:
        if st.button("🔄 Flip Board", use_container_width=True):
            st.session_state.player_color = not st.session_state.player_color
            st.rerun()
    
    st.markdown("---")
    
    # Export/Import
    st.subheader("💾 Save/Load")
    
    # Export PGN
    if st.button("📥 Export PGN", use_container_width=True):
        game = chess.pgn.Game()
        game.headers["Event"] = "NNUE Engine Game"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "You" if st.session_state.player_color else "Engine"
        game.headers["Black"] = "Engine" if st.session_state.player_color else "You"
        
        node = game
        board = chess.Board()
        for move_data in st.session_state.move_history:
            move = chess.Move.from_uci(move_data['uci'])
            node = node.add_variation(move)
            board.push(move)
        
        pgn_string = str(game)
        st.download_button(
            label="💾 Download PGN",
            data=pgn_string,
            file_name=f"chess_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn",
            mime="text/plain"
        )
    
    # Load FEN
    fen_input = st.text_input(
        "📋 Load Position (FEN)",
        placeholder="Paste FEN string..."
    )
    if st.button("⬆️ Load FEN", use_container_width=True) and fen_input:
        try:
            st.session_state.board = chess.Board(fen_input)
            st.session_state.move_history = []
            st.session_state.eval_history = []
            st.success("✅ Position loaded!")
            st.rerun()
        except:
            st.error("❌ Invalid FEN string!")
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.move_history:
        st.subheader("📊 Game Stats")
        st.metric("Moves Played", len(st.session_state.move_history))
        if st.session_state.eval_history:
            current_eval = st.session_state.eval_history[-1]
            st.metric("Current Eval", f"{current_eval:+.2f} cp")


# ============================================================================
# Main Content
# ============================================================================

# Title
st.title("♟️ NNUE Chess Engine")
st.markdown("Play against your trained neural network chess engine!")

# Check if engine is loaded
if not ENGINE_AVAILABLE or st.session_state.evaluator is None:
    st.error("⚠️ Engine not loaded! Make sure model files are present.")
    st.stop()

# Engine's first move if playing as Black
if st.session_state.game_started and not st.session_state.player_color and len(st.session_state.board.move_stack) == 0:
    with st.spinner("🤔 Engine thinking..."):
        make_engine_move(engine_depth, time_limit)
    st.session_state.game_started = False
    st.rerun()

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Chess Board")
    
    # Display board
    last_move = st.session_state.board.peek() if st.session_state.board.move_stack else None
    check_square = st.session_state.board.king(st.session_state.board.turn) if st.session_state.board.is_check() else None
    
    svg = board_to_svg(st.session_state.board, last_move, check_square)
    st.markdown(svg_to_html(svg), unsafe_allow_html=True)
    
    # Game status
    status = get_game_status(st.session_state.board)
    if "Checkmate" in status or "Draw" in status:
        st.success(status)
    elif "Check" in status:
        st.warning(status)
    else:
        st.info(status)
    
    # Current FEN
    with st.expander("📋 Position FEN"):
        st.code(st.session_state.board.fen(), language="text")
    
    # Move Input
    if not st.session_state.board.is_game_over():
        st.markdown("---")
        st.subheader("Your Move")
        
        # Show legal moves
        legal_moves = list(st.session_state.board.legal_moves)
        legal_moves_san = [format_move(m, st.session_state.board) for m in legal_moves]
        
        with st.expander(f"📝 Legal Moves ({len(legal_moves)})"):
            st.write(" • ".join(legal_moves_san))
        
        col_a, col_b = st.columns([3, 1])
        
        with col_a:
            move_input = st.text_input(
                "Enter move (UCI format, e.g., 'e2e4' or 'e7e8q' for promotion):",
                key="move_input",
                placeholder="e2e4"
            )
        
        with col_b:
            make_move_btn = st.button("▶️ Make Move", use_container_width=True)
        
        if make_move_btn and move_input:
            try:
                # Parse move
                move = chess.Move.from_uci(move_input.lower().strip())
                
                if move in st.session_state.board.legal_moves:
                    # Player move
                    san_move = format_move(move, st.session_state.board)
                    st.session_state.board.push(move)
                    
                    # Evaluate position
                    if st.session_state.evaluator:
                        eval_score = st.session_state.evaluator.evaluate(st.session_state.board)
                        st.session_state.eval_history.append(-eval_score)  # Flip for opponent
                    else:
                        eval_score = 0
                    
                    st.session_state.move_history.append({
                        'move': san_move,
                        'uci': move.uci(),
                        'evaluation': -eval_score,
                        'depth': 0,
                        'nodes': 0,
                        'time': 0
                    })
                    
                    # Check if game over
                    if not st.session_state.board.is_game_over():
                        # Engine response
                        with st.spinner("🤔 Engine thinking..."):
                            engine_move, score, stats = make_engine_move(engine_depth, time_limit)
                        
                        if engine_move:
                            st.success(f"Engine played: {format_move(engine_move, st.session_state.board.copy())}")
                    
                    st.rerun()
                else:
                    st.error("❌ Illegal move!")
            except:
                st.error("❌ Invalid move format! Use UCI notation (e.g., 'e2e4')")

with col2:
    st.subheader("Game Information")
    
    # Move History
    if st.session_state.move_history:
        st.markdown("### 📜 Move History")
        
        # Create move list
        moves_df = []
        for i, move_data in enumerate(st.session_state.move_history):
            move_num = (i // 2) + 1
            if i % 2 == 0:  # White's move
                moves_df.append({
                    'Move': move_num,
                    'White': move_data['move'],
                    'Black': '',
                    'Eval': f"{move_data['evaluation']:+.2f}"
                })
            else:  # Black's move
                moves_df[-1]['Black'] = move_data['move']
        
        st.dataframe(
            pd.DataFrame(moves_df),
            hide_index=True,
            use_container_width=True
        )
        
        # Last move details
        if st.session_state.move_history:
            last_move_data = st.session_state.move_history[-1]
            st.markdown("### 🔍 Last Move Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Evaluation", f"{last_move_data['evaluation']:+.2f} cp")
                st.metric("Depth", last_move_data['depth'])
            with col_b:
                st.metric("Nodes", f"{last_move_data['nodes']:,}")
                st.metric("Time", f"{last_move_data['time']:.2f}s")
    
    # Evaluation Graph
    if show_analysis and st.session_state.eval_history:
        st.markdown("### 📈 Evaluation History")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(st.session_state.eval_history))),
            y=st.session_state.eval_history,
            mode='lines+markers',
            name='Evaluation',
            line=dict(color='royalblue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            xaxis_title="Move Number",
            yaxis_title="Evaluation (centipawns)",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode='x unified'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick Analysis Button
    if not st.session_state.board.is_game_over():
        st.markdown("---")
        if st.button("🔍 Analyze Position", use_container_width=True):
            with st.spinner("Analyzing position..."):
                engine = ChessEngine(
                    st.session_state.evaluator,
                    max_depth=engine_depth,
                    use_quiescence=True
                )
                
                best_move, score = engine.search(
                    st.session_state.board,
                    depth=engine_depth
                )
                
                if best_move:
                    st.success(f"**Best move:** {format_move(best_move, st.session_state.board)}")
                    st.info(f"**Evaluation:** {score:+.2f} centipawns")
                    st.caption(f"Searched {engine.stats.nodes_searched:,} nodes in {engine.stats.time_elapsed:.2f}s")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>♟️ NNUE Chess Engine | Powered by Alpha-Beta Search & Deep Learning</p>
    <p>Made with ❤️ and Python</p>
    </div>
    """,
    unsafe_allow_html=True
)