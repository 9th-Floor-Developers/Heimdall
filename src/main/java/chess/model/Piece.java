package chess.model;

import static chess.model.Color.NONE;
import static chess.model.PieceType.EMPTY;

public class Piece extends Space {
	public Piece(PieceType type, Color color, int file, int rank) {
		super(type, color, file, rank);
		
		if (type == null || type == EMPTY)
			throw new IllegalArgumentException("Piece must have a real PieceType, got: " + type);
		if (color == null || color == NONE)
			throw new IllegalArgumentException("Piece must have a real Color, got: " + color);
	}
	
	@Override
	public Piece clone() {
		return (Piece) super.clone();
	}
	
	@Override
	public boolean isEmpty() {
		return false;
	}
	
	@Override
	public void setType(PieceType type) {
		super.setType(type);
		if (type == null || type == EMPTY)
			throw new IllegalArgumentException("Piece must have a real PieceType, got: " + type);
	}
	
	@Override
	public void setColor(Color color) {
		super.setColor(color);
		if (color == null || color == NONE)
			throw new IllegalArgumentException("Piece must have a real Color, got: " + color);
	}
}
