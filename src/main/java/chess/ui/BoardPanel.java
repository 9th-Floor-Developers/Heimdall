package chess.ui;

import chess.Board;
import static chess.ChessUtils.BOARD_SIZE;
import chess.MoveGenerator;
import chess.model.Move;
import chess.model.PieceType;
import chess.model.Space;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;
import java.net.URL;
import java.util.HashMap;
import java.util.HashSet;

public final class BoardPanel extends JPanel {
	private Space selected;
	private final HashSet<Move> selectedMoves;
	private final HashSet<Space> selectedMoveSpaces;
	private final HashMap<String, Image> pieceImages = new HashMap<>();
	private Board game;
	private int squareSize;
	
	public BoardPanel() {
		selected = null;
		selectedMoves = new HashSet<>();
		selectedMoveSpaces = new HashSet<>();
		
		addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(MouseEvent e) {
				handleClick(e);
			}
		});
		
		loadImages();
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		
		squareSize = Math.min(getWidth(), getHeight()) / 8;
		
		drawBoard(g);
		
		if (game == null)
			return;
		
		drawHighlights(g);
		drawPieces(g);
		drawLabels(g);
	}
	
	private void loadImages() {
		for (chess.model.Color color : chess.model.Color.values()) {
			if (color == chess.model.Color.NONE)
				continue;
			
			for (PieceType type : PieceType.values()) {
				if (type == PieceType.EMPTY)
					continue;
				
				String key = color.toString().toLowerCase() + '-' + type.toString().toLowerCase(), path = "/pieces/" + key + ".png";
				URL resource = getClass().getResource(path);
				if (resource == null)
					throw new RuntimeException("Resource not found: " + path);
				try {
					pieceImages.put(key, ImageIO.read(resource));
				} catch (IOException e) {
					throw new RuntimeException("Failed to load piece image " + path, e);
				}
			}
		}
	}
	
	// region Drawing Methods
	
	private void drawBoard(Graphics g) {
		if (squareSize == 0)
			return;
		
				g.setColor(((row + col) % 2 == 0) ? Color.WHITE : Color.GRAY);
		for (int row = 0; row < BOARD_SIZE; row++) {
			for (int col = 0; col < BOARD_SIZE; col++) {
				g.fillRect(
						col * squareSize,
						row * squareSize,
						squareSize,
						squareSize
				);
			}
		}
	}
	
	private void drawLabels(Graphics g) {
		if (squareSize == 0)
			return;
		
		g.setFont(g.getFont().deriveFont(Font.BOLD, 18f));
		FontMetrics fm = g.getFontMetrics();
		g.setColor(Color.BLACK);
		
		for (int i = 0; i < BOARD_SIZE; i++) {
			// files a-h along the bottom edge
			char file = (char) ('a' + i);
			g.drawString(
					String.valueOf(file),
					i * squareSize + 4,
					BOARD_SIZE * squareSize - 4
			);
			
			// ranks 8-1 down the left edge
			int rank = BOARD_SIZE - i;
			g.drawString(
					String.valueOf(rank),
					4,
					i * squareSize + fm.getAscent()
			);
		}
	}
	
	private void drawHighlights(Graphics g) {
		if (selected == null || squareSize == 0)
			return;
		
		// selected piece space
		g.setColor(Color.RED);
		g.fillRect(
				selected.getFile() * squareSize,
				(7 - selected.getRank()) * squareSize,
				squareSize, squareSize
		);
		
		// selected piece possible spaces
		g.setColor(Color.BLUE);
		for (Space space : selectedMoveSpaces)
			g.fillRect(
					space.getFile() * squareSize,
					(7 - space.getRank()) * squareSize,
					squareSize, squareSize
			);
	}
	
	private void drawPieces(Graphics g) {
		if (squareSize == 0)
			return;
		
		for (Space piece : game.getPieces()) {
			String key = piece.getColor().toString().toLowerCase() + "-"
					+ piece.getType().toString().toLowerCase();
			Image img = pieceImages.get(key);
			
			if (img == null)
				throw new RuntimeException("Piece " + key + " not found!");
			
			int drawRow = 7 - piece.getRank();
			g.drawImage(
					img,
					piece.getFile() * squareSize,
					drawRow * squareSize,
					squareSize, squareSize,
					this
			);
		}
	}
	
	// endregion
	
	private void handleClick(MouseEvent e) {
		if (game == null || squareSize == 0)
			return;
		
		int row = 7 - (e.getY() / squareSize),
				col = e.getX() / squareSize;
		
		if (row < 0 || row >= 8 || col < 0 || col >= 8)
			return;
		
		Space clicked = game.pieceAt(row, col);
		
		if (selected != null && selectedMoveSpaces.contains(clicked)) {
			game.makeMove(selectedMoves.stream()
					.filter(m -> game.pieceAt(m.to()).equals(clicked))
					.findFirst()
					.orElseThrow());
			clearSelection();
		} else if (clicked.getType() != PieceType.EMPTY) {
			clearSelection();
			selected = clicked;
			HashSet<Move> moves = MoveGenerator.generateLegalMoves(game, selected);
			selectedMoves.addAll(moves);
			selectedMoveSpaces.addAll(game.moveToSpace(moves));
		} else
			clearSelection();
		
		repaint();
	}
	
	private void clearSelection() {
		selected = null;
		selectedMoves.clear();
		selectedMoveSpaces.clear();
	}
	
	public void setGame(Board game) {
		this.game = game;
	}
}
