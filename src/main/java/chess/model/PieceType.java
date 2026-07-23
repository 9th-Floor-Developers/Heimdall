package chess.model;

public enum PieceType {
	EMPTY('.', 0),
	PAWN('p', 1),
	ROOK('r', 5),
	KNIGHT('n', 3),
	BISHOP('b', 3),
	QUEEN('q', 9),
	KING('k', 0);
	
	private final char letter;
	private final int material;
	
	PieceType(char letter, int material) {
		this.letter = letter;
		this.material = material;
	}
	
	public char getLetter() {
		return letter;
	}
	
	public int getMaterial() {
		return material;
	}
}
