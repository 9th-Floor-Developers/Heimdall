package chess;
import static chess.ChessUtils.*;
import chess.model.Color;
import static chess.model.Color.BLACK;
import static chess.model.Color.WHITE;
import chess.model.Move;
import static chess.model.PieceType.*;
import chess.model.Space;

import java.util.ArrayList;

public final class MoveGenerator {
	public static ArrayList<Move> generateLegalMoves(Board board) {
		ArrayList<Move> legal = new ArrayList<>(), pseudoLegal = generatePseudoLegalMoves(board);
		boolean white = board.isWhiteToMove();
		
		for (Move move : pseudoLegal) {
			board.makeMove(move);
			
			if (!board.isInCheck(white ? WHITE : BLACK))
				legal.add(move);
			
			board.undoMove();
		}
		
		return legal;
	}
	
	public static ArrayList<Move> generatePseudoLegalMoves(Board board) {
		ArrayList<Move> moves = new ArrayList<>();
		boolean white = board.isWhiteToMove();
		
		for (int i = 0; i < 64; i++) {
			Space piece = board.pieceAt(i);
			
			if (piece.isEmpty() || (piece.getColor() == WHITE) != white)
				continue;
			
			switch (piece.getType()) {
				case PAWN -> generatePawnMoves(board, i, white, moves);
				case KNIGHT -> generateKnightMoves(board, i, white, moves);
				case BISHOP -> generateSlidingMoves(board, i, white, moves, DIAG_DIRS);
				case ROOK -> generateSlidingMoves(board, i, white, moves, ORTHO_DIRS);
				case QUEEN -> generateSlidingMoves(board, i, white, moves, ALL_DIRS);
				case KING -> generateKingMoves(board, i, white, moves);
			}
		}
		
		return moves;
	}
	
	// region Piece Move Generations
	
	private static void generatePawnMoves(Board board, int index, boolean white,
	                                      ArrayList<Move> moves) {
		int f = fileOf(index), r = rankOf(index);
		int dir = white ? 1 : -1;
		int startRank = white ? 1 : 6;
		int promoRank = white ? 7 : 0;
		
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
	}
	
	private static void addPawnMoveWithPromotion(int from, int to, boolean isPromotion,
	                                             ArrayList<Move> moves) {
		if (isPromotion) {
			moves.add(new Move(from, to, QUEEN));
			moves.add(new Move(from, to, ROOK));
			moves.add(new Move(from, to, BISHOP));
			moves.add(new Move(from, to, KNIGHT));
		} else
			moves.add(new Move(from, to));
	}
	
	private static void generateKnightMoves(Board board, int sq, boolean white,
	                                        ArrayList<Move> moves) {
		int f = fileOf(sq), r = rankOf(sq);
		for (int[] dir : KNIGHT_DIRS) {
			int nf = f + dir[0], nr = r + dir[1];
			
			if (!onBoard(nf, nr))
				continue;
			
			int to = indexOf(nf, nr);
			Space toPiece = board.pieceAt(to);
			
			if (toPiece.isEmpty() || toPiece.getColor() == (white ? BLACK : WHITE))
				moves.add(new Move(sq, to));
		}
	}
	
	private static void generateSlidingMoves(Board board, int sq, boolean white,
	                                         ArrayList<Move> moves, int[][] dirs) {
		int f = fileOf(sq), r = rankOf(sq);
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
	}
	
	private static void generateKingMoves(Board board, int sq, boolean white, ArrayList<Move> moves) {
		Space piece = board.pieceAt(sq);
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
