package chess.model;

public enum PieceType {
	EMPTY('.'),
	PAWN('p'),
	ROOK('r'),
	KNIGHT('n'),
	BISHOP('b'),
	QUEEN('q'),
	KING('k');
	
	private final char letter;
	
	PieceType(char letter) {
		this.letter = letter;
	}
	
	public char getLetter() {
		return letter;
	}
}
