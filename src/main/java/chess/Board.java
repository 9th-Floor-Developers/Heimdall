package chess;

import static chess.ChessUtils.*;
import chess.model.*;
import static chess.model.Color.BLACK;
import static chess.model.Color.WHITE;
import static chess.model.PieceType.*;

import java.util.HashSet;
import java.util.Stack;

public final class Board implements Cloneable {
	private Space[] spaces;
	private Stack<MoveState> history;
	private boolean whiteCanCastleKingside, whiteCanCastleQueenside,
			blackCanCastleKingside, blackCanCastleQueenside;
	private boolean whiteToMove;
	private int enPassantTarget, halfMoveClock, fullMoveNumber;
	
	public Board() {
		spaces = new Space[64];
		history = new Stack<>();
		
		for (int i = 0; i < spaces.length; i++)
			spaces[i] = new Space(i % BOARD_SIZE, i / BOARD_SIZE);
		
		PieceType[] backRank = { ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK };
		for (int i = 0; i < BOARD_SIZE; i++) {
			spaces[i] = new Space(backRank[i], WHITE, i, 0);
			spaces[i + BOARD_SIZE] = new Space(PAWN, WHITE, i, 1);
			spaces[i + BOARD_SIZE * 6] = new Space(PAWN, BLACK, i, 6);
			spaces[i + BOARD_SIZE * 7] = new Space(backRank[i], BLACK, i, 7);
		}
		
		whiteToMove = true;
		whiteCanCastleKingside = whiteCanCastleQueenside = true;
		blackCanCastleKingside = blackCanCastleQueenside = true;
		enPassantTarget = -1;
		halfMoveClock = 0;
		fullMoveNumber = 1;
	}
	
	@Override
	public Board clone() {
		try {
			Board copy = (Board) super.clone();
			
			copy.spaces = new Space[64];
			for (int i = 0; i < 64; i++)
				copy.spaces[i] = spaces[i].clone();
			
			copy.history = (Stack<MoveState>) history.clone();
			
			return copy;
		} catch (CloneNotSupportedException e) {
			throw new AssertionError(e);
		}
	}
	
	public void makeMove(Move move) {
		Space capturedSpace = spaces[move.to()];
		MoveState state = new MoveState(
				move, capturedSpace.getType(), capturedSpace.getColor(),
				whiteCanCastleKingside, whiteCanCastleQueenside,
				blackCanCastleKingside, blackCanCastleQueenside,
				enPassantTarget, halfMoveClock
		);
		history.push(state);
		
		Space piece = spaces[move.from()];
		PieceType type = piece.getType();
		Color color = piece.getColor();
		
		if (move.isEnPassant()) {
			int capturedPawnSquare = indexOf(fileOf(move.to()), rankOf(move.from()));
			spaces[capturedPawnSquare].setEmpty();
		}
		
		capturedSpace.setPiece(type, color);
		spaces[move.from()].setEmpty();
		
		if (move.promotionPiece() != EMPTY)
			capturedSpace.setPiece(move.promotionPiece(), color);
		
		if (move.isCastle())
			moveCastleRook(move.to(), color);
		
		if (type == KING) {
			if (color == WHITE)
				whiteCanCastleKingside = whiteCanCastleQueenside = false;
			else if (color == BLACK)
				blackCanCastleKingside = blackCanCastleQueenside = false;
		}
		
		if (type == ROOK) {
			if (move.from() == indexOf(0, 0))
				whiteCanCastleQueenside = false;
			else if (move.from() == indexOf(7, 0))
				whiteCanCastleKingside = false;
			else if (move.from() == indexOf(0, 7))
				blackCanCastleQueenside = false;
			else if (move.from() == indexOf(7, 7))
				blackCanCastleKingside = false;
		}
		
		if (state.capturedPieceType() == ROOK) {
			if (move.to() == indexOf(0, 0))
				whiteCanCastleQueenside = false;
			else if (move.to() == indexOf(7, 0))
				whiteCanCastleKingside = false;
			else if (move.to() == indexOf(0, 7))
				blackCanCastleQueenside = false;
			else if (move.to() == indexOf(7, 7))
				blackCanCastleKingside = false;
		}
		
		enPassantTarget = (type == PAWN && Math.abs(rankOf(move.to()) - rankOf(move.from())) == 2)
				? indexOf(fileOf(move.from()), (rankOf(move.from()) + rankOf(move.to())) / 2)
				: -1;
		
		halfMoveClock = (type == PAWN || state.capturedPieceType() != EMPTY) ? 0 : halfMoveClock + 1;
		
		if (color == BLACK)
			fullMoveNumber++;
		
		whiteToMove = !whiteToMove;
	}
	
	public void undoMove() {
		if (history.isEmpty())
			return;
		
		MoveState state = history.pop();
		Move move = state.move();
		
		whiteToMove = !whiteToMove;
		
		int fromIndex = move.from();
		int toIndex = move.to();
		Space movedPiece = spaces[toIndex];
		
		if (move.promotionPiece() != EMPTY)
			movedPiece.setType(PAWN);
		
		spaces[fromIndex].setPiece(movedPiece.getType(), movedPiece.getColor());
		spaces[toIndex].setPiece(state.capturedPieceType(), state.capturedColor());
		
		if (move.isEnPassant()) {
			int capturedIndex = indexOf(fileOf(toIndex), rankOf(fromIndex));
			spaces[toIndex].setEmpty();
			spaces[capturedIndex].setPiece(PAWN, toMoveColor().getOpposite());
		}
		
		if (move.isCastle())
			undoCastleRook(move.to(), movedPiece.getColor());
		
		whiteCanCastleKingside = state.whiteCanCastleKingside();
		whiteCanCastleQueenside = state.whiteCanCastleQueenside();
		blackCanCastleKingside = state.blackCanCastleKingside();
		blackCanCastleQueenside = state.blackCanCastleQueenside();
		enPassantTarget = state.enPassantTarget();
		halfMoveClock = state.halfMoveClock();
		
		if (movedPiece.getColor() == BLACK)
			fullMoveNumber--;
	}
	
	public int evalBoard() {
		int score = 0;
		
		for (Space space : getPieces()) {
			PieceType type = space.getType();
			
			if (type == EMPTY || type == KING)
				continue;
			
			int index = indexOf(space.getFile(), space.getRank()),
					color = space.isWhite() ? 1 : -1;
			
			int material = type.getMaterial() + switch (type) {
				case PAWN -> PAWN_TABLE[index];
				case ROOK -> ROOK_TABLE[index];
				case KNIGHT -> KNIGHT_TABLE[index];
				case BISHOP -> BISHOP_TABLE[index];
				case QUEEN -> QUEEN_TABLE[index];
				default -> throw new IllegalStateException("Unexpected value: " + type);
			};
			
			score += material * color;
		}
		
		if (isEndgame()) {
			Space myKing = getKing(toMoveColor()),
					enemyKing = getKing(toMoveColor().getOpposite());
			int enemyKingCornerDist = distanceFromCenter(enemyKing);
			int kingDist = kingDistance(myKing, enemyKing);
			score += enemyKingCornerDist * 10 - kingDist * 4;
		}
		
		return score;
	}
	
	public void clear() {
		for (int i = 0; i < BOARD_SIZE; i++) {
			for (int j = 0; j < BOARD_SIZE; j++) {
				spaces[indexOf(i, j)] = new Space(i, j);
			}
		}
	}
	
	// region Helper Methods
	
	public boolean isSquareAttacked(Space square, Color byColor) {
		int f = square.getFile(), r = square.getRank();
		
		int pawnRankDir = (byColor == WHITE) ? -1 : 1;
		for (int df : new int[] { -1, 1 }) {
			int pf = f + df, pr = r + pawnRankDir;
			if (onBoard(pf, pr)) {
				Space s = pieceAt(pf, pr);
				if (s.getType() == PAWN && s.getColor() == byColor)
					return true;
			}
		}
		
		for (int[] off : KNIGHT_DIRS) {
			int nf = f + off[0], nr = r + off[1];
			if (onBoard(nf, nr)) {
				Space s = pieceAt(nf, nr);
				if (s.getType() == KNIGHT && s.getColor() == byColor)
					return true;
			}
		}
		
		for (int df = -1; df <= 1; df++) {
			for (int dr = -1; dr <= 1; dr++) {
				if (df == 0 && dr == 0)
					continue;
				
				int nf = f + df, nr = r + dr;
				if (onBoard(nf, nr)) {
					Space s = pieceAt(nf, nr);
					if (s.getType() == KING && s.getColor() == byColor)
						return true;
				}
			}
		}
		
		for (int[] dir : DIAG_DIRS) {
			int nf = f + dir[0], nr = r + dir[1];
			while (onBoard(nf, nr)) {
				Space s = pieceAt(nf, nr);
				PieceType type = s.getType();
				
				if (type != EMPTY) {
					if (s.getColor() == byColor && (type == BISHOP || type == QUEEN))
						return true;
					break;
				}
				
				nf += dir[0];
				nr += dir[1];
			}
		}
		
		for (int[] dir : ORTHO_DIRS) {
			int nf = f + dir[0], nr = r + dir[1];
			while (onBoard(nf, nr)) {
				Space s = pieceAt(nf, nr);
				PieceType type = s.getType();
				
				if (type != EMPTY) {
					if (s.getColor() == byColor && (type == ROOK || type == QUEEN))
						return true;
					break;
				}
				
				nf += dir[0];
				nr += dir[1];
			}
		}
		
		return false;
	}
	
	public Space pieceAt(int index) {
		return spaces[index];
	}
	
	public Space pieceAt(int file, int rank) {
		return spaces[indexOf(file, rank)];
	}
	
	public Space getKing(Color color) {
		for (Space piece : spaces)
			if (piece.getType() == KING && piece.getColor() == color)
				return piece;
		
		throw new IllegalStateException("No King Found For " + color.toString());
	}
	
	public boolean isInCheck(Color color) {
		Color enemy = (color == WHITE) ? BLACK : WHITE;
		return isSquareAttacked(getKing(color), enemy);
	}
	
	private void movePiece(int from, int to) {
		spaces[to].setPiece(spaces[from].getType(), spaces[from].getColor());
		spaces[from].setEmpty();
	}
	
	private void moveCastleRook(int kingTo, Color color) {
		int rank = color == WHITE ? 0 : 7;
		
		if (kingTo == indexOf(6, rank))
			movePiece(indexOf(7, rank), indexOf(5, rank));
		else if (kingTo == indexOf(2, rank))
			movePiece(indexOf(0, rank), indexOf(3, rank));
	}
	
	private void undoCastleRook(int kingTo, Color color) {
		int rank = color == WHITE ? 0 : 7;
		
		if (kingTo == indexOf(6, rank))
			movePiece(indexOf(5, rank), indexOf(7, rank));
		else if (kingTo == indexOf(2, rank))
			movePiece(indexOf(3, rank), indexOf(0, rank));
	}
	
	public HashSet<Space> getPieces() {
		HashSet<Space> map = new HashSet<>();
		
		for (Space space : spaces) {
			if (space.getType() == EMPTY)
				continue;
			map.add(space);
		}
		
		return map;
	}
	
	public boolean isEndgame() {
		int totalMaterial = 0;
		for (Space piece : getPieces()) {
			if (piece.getType() != PieceType.KING)
				totalMaterial += piece.getType().getMaterial();
		}
		return totalMaterial <= 20;
	}
	
	private static int distanceFromCenter(Space space) {
		int file = space.getFile(), rank = space.getRank(),
				fileDist = Math.max(3 - file, file - 4),
				rankDist = Math.max(3 - rank, rank - 4);
		return fileDist + rankDist;
	}
	
	private static int kingDistance(Space king1, Space king2) {
		int file1 = king1.getFile(), file2 = king2.getFile(),
				rank1 = king1.getRank(), rank2 = king2.getRank();
		return Math.max(Math.abs(file1 - file2), Math.abs(rank1 - rank2));  // Chebyshev distance
	}
	
	public HashSet<Space> moveToSpace(HashSet<Move> moves) {
		HashSet<Space> spaces = new HashSet<>();
		for (Move move : moves)
			spaces.add(pieceAt(move.to()));
		return spaces;
	}
	
	public Color toMoveColor() {
		return whiteToMove ? WHITE : BLACK;
	}
	
	// endregion
	
	// region Getters/Setters
	public Space[] getSpaces() {
		return spaces;
	}
	
	public void setSpace(int file, int rank, Space space) {
		spaces[indexOf(file, rank)] = space;
	}
	
	public boolean isWhiteCanCastleKingside() {
		return whiteCanCastleKingside;
	}
	
	public void setWhiteCanCastleKingside(boolean whiteCanCastleKingside) {
		this.whiteCanCastleKingside = whiteCanCastleKingside;
	}
	
	public boolean isWhiteCanCastleQueenside() {
		return whiteCanCastleQueenside;
	}
	
	public void setWhiteCanCastleQueenside(boolean whiteCanCastleQueenside) {
		this.whiteCanCastleQueenside = whiteCanCastleQueenside;
	}
	
	public boolean isBlackCanCastleKingside() {
		return blackCanCastleKingside;
	}
	
	public void setBlackCanCastleKingside(boolean blackCanCastleKingside) {
		this.blackCanCastleKingside = blackCanCastleKingside;
	}
	
	public boolean isBlackCanCastleQueenside() {
		return blackCanCastleQueenside;
	}
	
	public void setBlackCanCastleQueenside(boolean blackCanCastleQueenside) {
		this.blackCanCastleQueenside = blackCanCastleQueenside;
	}
	
	public boolean isWhiteToMove() {
		return whiteToMove;
	}
	
	public Color getTurnColor() {
		return whiteToMove ? Color.WHITE : Color.BLACK;
	}
	
	public Color getOppositeColor() {
		return whiteToMove ? Color.BLACK : WHITE;
	}
	
	public void setWhiteToMove(boolean whiteToMove) {
		this.whiteToMove = whiteToMove;
	}
	
	public int getEnPassantTarget() {
		return enPassantTarget;
	}
	
	public void setEnPassantTarget(int enPassantTarget) {
		this.enPassantTarget = enPassantTarget;
	}
	
	public int getHalfMoveClock() {
		return halfMoveClock;
	}
	
	public void setHalfMoveClock(int halfMoveClock) {
		this.halfMoveClock = halfMoveClock;
	}
	
	public int getFullMoveNumber() {
		return fullMoveNumber;
	}
	
	public void setFullMoveNumber(int fullMoveNumber) {
		this.fullMoveNumber = fullMoveNumber;
	}
	
	public Stack<MoveState> getHistory() {
		return history;
	}
	
	// endregion
}
