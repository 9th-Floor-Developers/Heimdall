package chess.model;

import static chess.model.Color.NONE;
import static chess.model.PieceType.EMPTY;

public class Space implements Cloneable {
	private final int file, rank;
	private PieceType type;
	private Color color;
	
	public Space(int file, int rank) {
		type = EMPTY;
		color = NONE;
		this.file = file;
		this.rank = rank;
	}
	
	public Space(PieceType type, Color color, int file, int rank) {
		this.type = type;
		this.color = color;
		this.file = file;
		this.rank = rank;
	}
	
	@Override
	public Space clone() {
		try {
			return (Space) super.clone();
		} catch (CloneNotSupportedException e) {
			throw new AssertionError();
		}
	}
	
	public boolean isEmpty() {
		return type == EMPTY;
	}
	
	public void setEmpty() {
		type = EMPTY;
		color = NONE;
	}
	
	public void setPiece(PieceType type, Color color) {
		this.type = type;
		this.color = color;
	}
	
	// region Getters/Setters
	
	public PieceType getType() {
		return type;
	}
	
	public void setType(PieceType type) {
		this.type = type;
	}
	
	public Color getColor() {
		return color;
	}
	
	public void setColor(Color color) {
		this.color = color;
	}
	
	public int getFile() {
		return file;
	}
	
	public int getRank() {
		return rank;
	}
	
	// endregion
}
