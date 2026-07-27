package chess.ui;

import chess.Board;
import chess.MoveGenerator;
import chess.model.Move;
import chess.model.PieceType;
import chess.model.Space;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashSet;

public class BoardPanel extends JPanel {
	private Space selected;
	private final HashSet<Move> selectedMoves;
	private final HashSet<Space> selectedMoveSpaces;
	private Board game;
	
	public BoardPanel() {
		selected = null;
		selectedMoves = new HashSet<>();
		selectedMoveSpaces = new HashSet<>();
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		
		drawBoard(g);
		
		addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				handleClick(e);
			}
		});
		
		drawHighlights(g);
		
		// draw pieces
	}
	
	private void drawBoard(Graphics g) {
		int squareSize = getWidth() / 8;
		for (int row = 0; row < 8; row++) {
			for (int col = 0; col < 8; col++) {
				g.setColor(((row + col) % 2 == 0) ? Color.WHITE : Color.GRAY);
				
				g.fillRect(
						col * squareSize,
						row * squareSize,
						squareSize,
						squareSize
				);
			}
		}
	}
	
	private void drawHighlights(Graphics g) {
		if (selected == null) {
			return;
		}
		
		int squareSize = getWidth() / 8;
		
		// selected piece space
		g.setColor(Color.RED);
		g.drawRect(
				selected.getFile() * squareSize,
				selected.getRank() * squareSize,
				squareSize, squareSize
		);
		
		// selected piece possible spaces
		g.setColor(Color.BLUE);
		for (Space space : selectedMoveSpaces)
			g.drawRect(
					space.getFile() * squareSize,
					space.getRank() * squareSize,
					squareSize, squareSize
			);
	}
	
	private void handleClick(MouseEvent e) {
		int squareSize = getWidth() / 8,
				row = e.getY() / squareSize,
				col = e.getX() / squareSize;
		
		if (row < 0 || row >= 8 || col < 0 || col >= 8)
			return;
		
		Space clicked = game.pieceAt(row, col);
		
		if (selected == null) {  // clicking any square
			if (clicked.getType() != PieceType.EMPTY) {  // clicking any piece
				selected = clicked;
				HashSet<Move> moves = MoveGenerator.generateLegalMoves(game, selected);
				selectedMoves.addAll(moves);
				selectedMoveSpaces.addAll(game.moveToSpace(moves));
				repaint();
			}
		} else {
			if (selectedMoveSpaces.contains(clicked)) {  // clicking a possible move space
//				Move chosen = selectedMoves.stream()
//						.filter(m -> m.to() == (clicked))
//						.findFirst()
//						.orElseThrow();
//				game.makeMove(chosen);
				selected = null;
				selectedMoves.clear();
				selectedMoveSpaces.clear();
				repaint();
			} else {  // clicking off a piece
				selected = null;
				selectedMoves.clear();
				selectedMoveSpaces.clear();
				repaint();
			}
		}
	}
	
	public void setGame(Board game) {
		this.game = game;
	}
}
