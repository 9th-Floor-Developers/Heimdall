package chess.ui;

import chess.Board;
import static chess.ChessUtils.BOARD_SIZE;
import chess.MoveGenerator;
import chess.model.Color;
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
import java.util.EnumMap;
import java.util.HashSet;

public final class BoardPanel extends JPanel {
	private Space selected;
	private final HashSet<Move> selectedMoves;
	private final HashSet<Space> selectedMoveSpaces;
	private final EnumMap<Color, EnumMap<PieceType, Image>> pieceImages;
	private Board game;
	
	public BoardPanel() {
		selected = null;
		selectedMoves = new HashSet<>();
		selectedMoveSpaces = new HashSet<>();
		pieceImages = new EnumMap<>(Color.class);
		
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
		
		int squareSize = squareSize();
		
		drawBoard(g, squareSize);
		
		if (game == null)
			return;
		
		drawHighlights(g, squareSize);
		drawPieces(g, squareSize);
		drawLabels(g, squareSize);
	}
	
	private int squareSize() {
		return Math.min(getWidth(), getHeight()) / BOARD_SIZE;
	}
	
	private void loadImages() {
		for (Color color : Color.values()) {
			if (color == Color.NONE)
				continue;
			
			EnumMap<PieceType, Image> typeMap = new EnumMap<>(PieceType.class);
			for (PieceType type : PieceType.values()) {
				if (type == PieceType.EMPTY)
					continue;
				
				String key = color.toString().toLowerCase() + '-' + type.toString().toLowerCase(),
						path = "/pieces/" + key + ".png";
				URL resource = getClass().getResource(path);
				if (resource == null)
					throw new RuntimeException("Missing image resource: " + path);
				
				try {
					typeMap.put(type, ImageIO.read(resource));
				} catch (IOException e) {
					throw new RuntimeException("Failed to load image: " + path, e);
				}
			}
			
			pieceImages.put(color, typeMap);
		}
	}
	
	// region Draw Methods
	
	private void drawBoard(Graphics g, int squareSize) {
		for (int row = 0; row < BOARD_SIZE; row++) {
			for (int col = 0; col < BOARD_SIZE; col++) {
				g.setColor(((row + col) % 2 == 0) ? java.awt.Color.WHITE : java.awt.Color.GRAY);
				g.fillRect(
						col * squareSize,
						row * squareSize,
						squareSize,
						squareSize
				);
			}
		}
	}
	
	private void drawLabels(Graphics g, int squareSize) {
		g.setFont(g.getFont().deriveFont(Font.BOLD, 18f));
		FontMetrics fm = g.getFontMetrics();
		g.setColor(java.awt.Color.BLACK);
		
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
	
	private void drawHighlights(Graphics g, int squareSize) {
		if (selected == null)
			return;
		
		// selected piece space
		g.setColor(java.awt.Color.RED);
		g.fillRect(
				selected.getFile() * squareSize,
				(7 - selected.getRank()) * squareSize,
				squareSize, squareSize
		);
		
		// selected piece possible spaces
		g.setColor(java.awt.Color.BLUE);
		for (Space space : selectedMoveSpaces)
			g.fillRect(
					space.getFile() * squareSize,
					(7 - space.getRank()) * squareSize,
					squareSize, squareSize
			);
	}
	
	private void drawPieces(Graphics g, int squareSize) {
		for (Space piece : game.getPieces()) {
			Image img = pieceImages.get(piece.getColor()).get(piece.getType());
			
			if (img == null)
				throw new RuntimeException("Piece image not found for " + piece.getColor() + " " + piece.getType());
			
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
		if (game == null)
			return;
		
		int squareSize = squareSize();
		if (squareSize == 0)
			return;
		
		int row = 7 - (e.getY() / squareSize),
				col = e.getX() / squareSize;
		
		if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE)
			return;
		
		Space clicked = game.pieceAt(col, row);
		
		if (selected != null && selectedMoveSpaces.contains(clicked)) {
			// complete the move to the clicked destination
			game.makeMove(
					selectedMoves.stream()
							.filter(m -> game.pieceAt(m.to()).equals(clicked))
							.findFirst()
							.orElseThrow());
			clearSelection();
		} else if (clicked.getType() != PieceType.EMPTY
				&& clicked.getColor() == (game.isWhiteToMove() ? Color.WHITE : Color.BLACK)) {
			// select a new piece
			clearSelection();
			selected = clicked;
			HashSet<Move> moves = MoveGenerator.generateLegalMoves(game, selected);
			selectedMoves.addAll(moves);
			selectedMoveSpaces.addAll(game.moveToSpace(moves));
		} else
			// clicked an empty, non-destination square
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
		repaint();
	}
}
