package chess.model;

import static chess.ChessUtils.*;
import static chess.model.PieceType.EMPTY;

public record Move(int from,
                   int to,
                   PieceType promotionPiece,
                   boolean isCastle,
                   boolean isEnPassant) {
	
	public Move(int from, int to) {
		this(from, to, EMPTY, false, false);
	}
	
	public Move(int from, int to, PieceType promotionPiece) {
		this(from, to, promotionPiece, false, false);
	}
	
	public String toLongAlgebraic() {
		Space fromSpace = new Space(fileOf(from), rankOf(from)),
				toSpace = new Space(fileOf(to), rankOf(to));
		String algebraic = squareName(fromSpace) + squareName(toSpace);
		
		if (promotionPiece != EMPTY)
			algebraic += promotionPiece.getLetter();
		
		return algebraic;
	}
}
