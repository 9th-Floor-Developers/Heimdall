package chess;
import static chess.ChessUtils.*;
import chess.model.Color;
import static chess.model.Color.BLACK;
import static chess.model.Color.WHITE;
import chess.model.Move;
import static chess.model.PieceType.*;
import chess.model.Space;

import java.util.HashSet;

public final class MoveGenerator {
	public static HashSet<Move> generateLegalMoves(Board board) {
		HashSet<Move> legal = new HashSet<>(), pseudoLegal = generatePseudoLegalMoves(board);
		boolean white = board.isWhiteToMove();
		
		for (Move move : pseudoLegal) {
			board.makeMove(move);
			
			if (!board.isInCheck(white ? WHITE : BLACK))
				legal.add(move);
			
			board.undoMove();
		}
		
		return legal;
	}
	
	public static HashSet<Move> generateLegalMoves(Board board, Space piece) {
		boolean white = board.isWhiteToMove();
		HashSet<Move> legal = new HashSet<>(), pseudoLegal = generatePieceMoves(board, piece, white);
		
		for (Move move : pseudoLegal) {
			board.makeMove(move);
			
			if (!board.isInCheck(white ? WHITE : BLACK))
				legal.add(move);
			
			board.undoMove();
		}
		
		return legal;
	}
	
	public static HashSet<Move> generatePseudoLegalMoves(Board board) {
		HashSet<Move> moves = new HashSet<>();
		boolean white = board.isWhiteToMove();
		
		for (Space piece : board.getPieces()) {
			if ((piece.getColor() == WHITE) != white)
				continue;
			
			moves.addAll(generatePieceMoves(board, piece, white));
		}
		
		return moves;
	}
	
	private static HashSet<Move> generatePieceMoves(Board board, Space piece, boolean white) {
		if (piece.isEmpty())
			return new HashSet<>();
		
		int index = indexOf(piece.getFile(), piece.getRank());
		return switch (piece.getType()) {
			case PAWN -> generatePawnMoves(board, index, white);
			case KNIGHT -> generateKnightMoves(board, index, white);
			case BISHOP -> generateSlidingMoves(board, index, white, DIAG_DIRS);
			case ROOK -> generateSlidingMoves(board, index, white, ORTHO_DIRS);
			case QUEEN -> generateSlidingMoves(board, index, white, ALL_DIRS);
			case KING -> generateKingMoves(board, index, white);
			default -> new HashSet<>();
		};
	}
	
	// region Piece Move Generations
	
	private static HashSet<Move> generatePawnMoves(Board board, int index, boolean white) {
		int f = fileOf(index), r = rankOf(index);
		int dir = white ? 1 : -1;
		int startRank = white ? 1 : 6;
		int promoRank = white ? 7 : 0;
		HashSet<Move> moves = new HashSet<>();
		
		// pushes
		int oneRank = r + dir;
		if (onBoard(f, oneRank) && board.pieceAt(indexOf(f, oneRank)).isEmpty()) {
			addPawnMoveWithPromotion(index, indexOf(f, oneRank), oneRank == promoRank, moves);
			
			// pawn double push
			int twoRank = r + 2 * dir;
			if (r == startRank && onBoard(f, twoRank) && board.pieceAt(indexOf(f, twoRank)).isEmpty())
				moves.add(new Move(index, indexOf(f, twoRank)));
		}
		
		// captures
		for (int df : new int[] { -1, 1 }) {
			int cf = f + df, cr = r + dir;
			
			if (!onBoard(cf, cr))
				continue;
			
			int to = indexOf(cf, cr);
			Space toPiece = board.pieceAt(to);
			
			if (toPiece.getType() != EMPTY && toPiece.getColor() == (white ? BLACK : WHITE))
				addPawnMoveWithPromotion(index, to, cr == promoRank, moves);
			else if (to == board.getEnPassantTarget())
				moves.add(new Move(index, to, EMPTY, false, true));
		}
		
		return moves;
	}
	
	private static void addPawnMoveWithPromotion(int from, int to, boolean isPromotion,
	                                             HashSet<Move> moves) {
		if (isPromotion) {
			moves.add(new Move(from, to, QUEEN));
			moves.add(new Move(from, to, ROOK));
			moves.add(new Move(from, to, BISHOP));
			moves.add(new Move(from, to, KNIGHT));
		} else
			moves.add(new Move(from, to));
	}
	
	private static HashSet<Move> generateKnightMoves(Board board, int sq, boolean white) {
		int f = fileOf(sq), r = rankOf(sq);
		HashSet<Move> moves = new HashSet<>();
		
		for (int[] dir : KNIGHT_DIRS) {
			int nf = f + dir[0], nr = r + dir[1];
			
			if (!onBoard(nf, nr))
				continue;
			
			int to = indexOf(nf, nr);
			Space toPiece = board.pieceAt(to);
			
			if (toPiece.isEmpty() || toPiece.getColor() == (white ? BLACK : WHITE))
				moves.add(new Move(sq, to));
		}
		
		return moves;
	}
	
	private static HashSet<Move> generateSlidingMoves(Board board, int sq, boolean white, int[][] dirs) {
		int f = fileOf(sq), r = rankOf(sq);
		HashSet<Move> moves = new HashSet<>();
		
		for (int[] dir : dirs) {
			int nf = f + dir[0], nr = r + dir[1];
			
			while (onBoard(nf, nr)) {
				int to = indexOf(nf, nr);
				Space toPiece = board.pieceAt(to);
				
				if (toPiece.isEmpty())
					moves.add(new Move(sq, to));
				else {
					if (toPiece.getColor() == (white ? BLACK : WHITE))
						moves.add(new Move(sq, to));
					break;
				}
				
				nf += dir[0];
				nr += dir[1];
			}
		}
		
		return moves;
	}
	
	private static HashSet<Move> generateKingMoves(Board board, int sq, boolean white) {
		Space piece = board.pieceAt(sq);
		HashSet<Move> moves = new HashSet<>();
		int f = piece.getFile(), r = piece.getRank();
		
		for (int df : new int[] { -1, 0, 1 }) {
			for (int dr : new int[] { -1, 0, 1 }) {
				if (df == 0 && dr == 0)
					continue;
				
				int nf = f + df, nr = r + dr;
				
				if (!onBoard(nf, nr))
					continue;
				
				int to = indexOf(nf, nr);
				Space toPiece = board.pieceAt(to);
				
				if (toPiece.isEmpty() || toPiece.getColor() == (white ? BLACK : WHITE))
					moves.add(new Move(sq, to));
			}
		}
		
		Color enemy = white ? BLACK : WHITE;
		boolean inCheck = board.isSquareAttacked(piece, enemy);
		
		if (!inCheck) {
			int rank = white ? 0 : 7;
			Color color = white ? BLACK : WHITE;
			
			if (canCastleKingSide(board, rank, color, enemy))
				moves.add(new Move(sq, indexOf(6, rank), EMPTY, true, false));
			if (canCastleQueenSide(board, rank, color, enemy))
				moves.add(new Move(sq, indexOf(2, rank), EMPTY, true, false));
		}
		
		return moves;
	}
	
	// endregion
	
	// region Castling Checks
	
	private static boolean canCastleKingSide(Board board, int rank, Color color, Color enemy) {
		boolean canCastleKingSide = color == WHITE
				? board.isWhiteCanCastleKingside()
				: board.isBlackCanCastleKingside();
		
		Space rookSpace = board.pieceAt(7, rank),
				rookToSpace = board.pieceAt(5, rank),
				kingToSpace = board.pieceAt(6, rank);
		
		return canCastleKingSide
				&& rookToSpace.isEmpty()
				&& kingToSpace.isEmpty()
				&& rookSpace.getType() == ROOK
				&& rookSpace.getColor() == color
				&& !board.isSquareAttacked(rookToSpace, enemy)
				&& !board.isSquareAttacked(kingToSpace, enemy);
	}
	
	private static boolean canCastleQueenSide(Board board, int rank, Color color, Color enemy) {
		boolean canCastleQueenSide = color == WHITE
				? board.isWhiteCanCastleQueenside()
				: board.isBlackCanCastleQueenside();
		
		Space rookFromSpace = board.pieceAt(0, rank),
				kingToSpace = board.pieceAt(2, rank),
				rookToSpace = board.pieceAt(3, rank),
				middleSpace = board.pieceAt(1, rank);
		
		return canCastleQueenSide
				&& rookToSpace.isEmpty()
				&& kingToSpace.isEmpty()
				&& middleSpace.isEmpty()
				&& rookFromSpace.getType() == ROOK
				&& rookFromSpace.getColor() == color
				&& !board.isSquareAttacked(rookToSpace, enemy)
				&& !board.isSquareAttacked(kingToSpace, enemy);
	}
	
	// endregion
}
