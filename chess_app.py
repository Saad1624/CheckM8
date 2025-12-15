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
    from alpha_beta.alpha_beta import NNUEEvaluator, ChessEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    st.error("alpha_beta.py not found! Please ensure it's in the same directory.")

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="NNUE Chess Engine",
    page_icon="♟",
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
    
    if 'evaluator' not in st.session_state and ENGINE_AVAILABLE:
        try:
            # Try ONNX first (faster)
            evaluator = NNUEEvaluator('models/nnue_model.onnx', use_onnx=True, use_advanced_features=True)
            st.session_state.evaluator = evaluator
        except:
            try:
                # Fallback to PyTorch
                evaluator = NNUEEvaluator('models/best_nnue_model.pth', use_onnx=False, use_advanced_features=True)
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
    
    if 'engine_move_pending' not in st.session_state:
        st.session_state.engine_move_pending = False
    
    if 'move_to_make' not in st.session_state:
        st.session_state.move_to_make = None

initialize_session_state()

# ============================================================================
# Helper Functions
# ============================================================================
def board_to_svg(board, last_move=None, check_square=None):
    """Convert board to SVG with highlighting."""
    fill = {}
    
    # Highlight last move
    if last_move:
        fill[last_move.from_square] = "#cdd26a"  # Yellow-green for origin square
        fill[last_move.to_square] = "#aaa23a"    # Darker yellow-green for destination
    
    # Highlight check
    if check_square is not None:
        fill[check_square] = "#ff6b6b"  # Red for check
    
    svg = chess.svg.board(
        board,
        fill=fill,  # <-- Changed from squares to fill
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
        return f"Checkmate! {winner} wins!"
    elif board.is_stalemate():
        return "Draw by stalemate"
    elif board.is_insufficient_material():
        return "Draw by insufficient material"
    elif board.can_claim_fifty_moves():
        return "Draw by 50-move rule"
    elif board.can_claim_threefold_repetition():
        return "Draw by repetition"
    elif board.is_check():
        return "Check!"
    else:
        return "Game in progress"

def make_engine_move(depth, time_limit=None):
    """Make engine move with progress indication."""
    if st.session_state.evaluator is None:
        st.error("Engine not available!")
        return None, None, None
    
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

def process_move(move_input):
    """Process the player's move."""
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
                st.session_state.engine_move_pending = True
            
            return True, None
        else:
            return False, "❌ Illegal move!"
    except Exception as e:
        return False, f"❌ Invalid move format! Use UCI notation (e.g., 'e2e4'). Error: {e}"

# ============================================================================
# Sidebar - Game Controls
# ============================================================================
with st.sidebar:
    st.title("♟ NNUE Chess Engine")
    st.markdown("---")
    
    # New Game Section
    st.subheader("New Game")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Play as White", use_container_width=True, key="play_white"):
            st.session_state.board = chess.Board()
            st.session_state.move_history = []
            st.session_state.eval_history = []
            st.session_state.player_color = chess.WHITE
            st.session_state.game_started = True
            st.session_state.engine_move_pending = False
            st.rerun()
    
    with col2:
        if st.button("Play as Black", use_container_width=True, key="play_black"):
            st.session_state.board = chess.Board()
            st.session_state.move_history = []
            st.session_state.eval_history = []
            st.session_state.player_color = chess.BLACK
            st.session_state.game_started = True
            st.session_state.engine_move_pending = True
            st.rerun()
    
    if st.button("Reset Board", use_container_width=True, key="reset"):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.eval_history = []
        st.session_state.game_started = False
        st.session_state.engine_move_pending = False
        st.rerun()
    
    st.markdown("---")
    
    # Engine Settings
    st.subheader("⚙️ Engine Settings")
    
    engine_depth = st.slider(
        "Search Depth",
        min_value=1,
        max_value=8,
        value=5,
        help="Higher = stronger but slower",
        key="depth_slider"
    )
    
    time_limit = st.slider(
        "Time Limit (seconds)",
        min_value=1,
        max_value=60,
        value=10,
        help="Maximum time per move",
        key="time_slider"
    )
    
    show_analysis = st.checkbox("Show Analysis", value=True, key="show_analysis")
    
    st.markdown("---")
    
    # Game Actions
    st.subheader("🎮 Game Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Undo Move", use_container_width=True, key="undo"):
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
                st.session_state.engine_move_pending = False
                st.rerun()
    
    with col2:
        if st.button("Flip Board", use_container_width=True, key="flip"):
            st.session_state.player_color = not st.session_state.player_color
            st.rerun()
    
    st.markdown("---")
    
    # Export/Import
    st.subheader("💾 Save/Load")
    
    # Export PGN
    if st.button("Export PGN", use_container_width=True, key="export_pgn"):
        game = chess.pgn.Game()
        game.headers["Event"] = "NNUE Engine Game"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "You" if st.session_state.player_color == chess.WHITE else "Engine"
        game.headers["Black"] = "Engine" if st.session_state.player_color == chess.WHITE else "You"
        
        node = game
        board = chess.Board()
        for move_data in st.session_state.move_history:
            move = chess.Move.from_uci(move_data['uci'])
            node = node.add_variation(move)
            board.push(move)
        
        pgn_string = str(game)
        st.download_button(
            label="Download PGN",
            data=pgn_string,
            file_name=f"chess_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn",
            mime="text/plain",
            key="download_pgn"
        )
    
    st.markdown("---")
    # Load FEN

    st.subheader("♟ FEN String")
    fen_input = st.text_input(
        "FEN string",
        placeholder="Paste FEN and press Enter...",
        key="fen_input_box"
        )
    
    # Check if FEN input changed (Enter was pressed)
    if 'last_fen_input' not in st.session_state:
        st.session_state.last_fen_input = ""
        
    if fen_input and fen_input != st.session_state.last_fen_input:
        try:
            # Validate and load
            test_board = chess.Board(fen_input)
            st.session_state.board = test_board
            st.session_state.move_history = []
            st.session_state.eval_history = []
            st.session_state.engine_move_pending = False
            st.session_state.last_fen_input = fen_input
            st.success("✅ Position loaded!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Invalid FEN")
            st.session_state.last_fen_input = fen_input

    if fen_input and fen_input != st.session_state.last_fen_input:
        try_load_fen(fen_input)
    
    # Load FEN Button
    if st.button("Load FEN", use_container_width=True, key="load_fen_btn"):
        if fen_input:
            try_load_fen(fen_input)
        else:
            st.warning("⚠️ Please enter a FEN string first")
    
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
st.title("♟ NNUE Chess Engine")
st.markdown("Play against your trained neural network chess engine!")

# Check if engine is loaded
if not ENGINE_AVAILABLE or st.session_state.evaluator is None:
    st.error("⚠️ Engine not loaded! Make sure model files are present.")
    st.stop()

# Engine's first move if playing as Black
if st.session_state.engine_move_pending and not st.session_state.board.is_game_over():
    with st.spinner("🤔 Engine thinking..."):
        make_engine_move(engine_depth, time_limit)
    st.session_state.engine_move_pending = False
    st.rerun()

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("♟ Chess Board")
    
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
    
    # Move Input - RIGHT BELOW THE BOARD
    if not st.session_state.board.is_game_over():
        st.markdown("---")
        st.subheader("✋ Your Move")
        
        # Show legal moves in compact format
        legal_moves = list(st.session_state.board.legal_moves)
        legal_moves_san = [format_move(m, st.session_state.board) for m in legal_moves]
        
        with st.expander(f"📖 Legal Moves ({len(legal_moves)})"):
            st.write(" • ".join(legal_moves_san))
        
        # Create a form for Enter key support
        with st.form(key="move_form", clear_on_submit=True):
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                move_input = st.text_input(
                    "Enter move (UCI format, e.g., 'e2e4' or 'e7e8q'):",
                    key="move_input_field",
                    placeholder="e2e4",
                    label_visibility="collapsed"
                )
            
            with col_b:
                submit_button = st.form_submit_button("Make Move", use_container_width=True)
            
            # Process move when form is submitted (Enter key or button click)
            if submit_button and move_input:
                success, error = process_move(move_input)
                if success:
                    st.rerun()
                else:
                    st.error(error)
    
    # Current FEN - moved below move input
    with st.expander("📋 Position FEN"):
        st.code(st.session_state.board.fen(), language="text")

with col2:
    st.subheader("ℹ️ Game Information")
    
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
                if moves_df:
                    moves_df[-1]['Black'] = move_data['move']
        
        st.dataframe(
            pd.DataFrame(moves_df),
            hide_index=True,
            use_container_width=True,
            height=300
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
        if st.button("🔍 Analyze Position", use_container_width=True, key="analyze_btn"):
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
    <p>♟ NNUE Chess Engine | Powered by Alpha-Beta Search & Deep Learning</p>
    <p>Made with Python 🐍</p>
    </div>
    """,
    unsafe_allow_html=True
)