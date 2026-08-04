package chess;

import static chess.ChessUtils.*;
import chess.model.Color;
import chess.model.PieceType;
import chess.model.Space;

public final class FenUtils {
	public static String exportFen(Board board) {
		StringBuilder fen = new StringBuilder();
		
		for (int rank = 7; rank >= 0; rank--) {
			int empty = 0;
			
			for (int file = 0; file < 8; file++) {
				Space piece = board.pieceAt(file, rank);
				
				if (piece.isEmpty())
					empty++;
				else {
					if (empty > 0) {
						fen.append(empty);
						empty = 0;
					}
					
					fen.append(pieceToFen(piece));
				}
			}
			
			if (empty > 0)
				fen.append(empty);
			
			if (rank != 0)
				fen.append('/');
		}
		
		fen.append(' ').append(board.isWhiteToMove() ? 'w' : 'b');
		
		String castling = getCastlingRights(board);
		fen.append(' ').append(castling.isEmpty() ? "-" : castling);
		
		String ep = getEnPassantSquare(board);
		fen.append(' ').append(ep == null ? "-" : ep);
		
		fen.append(' ').append(board.getHalfMoveClock());
		fen.append(' ').append(board.getFullMoveNumber());
		
		return fen.toString();
	}
	
	public static void importFen(Board board, String fen) {
		String[] parts = fen.trim().split("\\s+");
		
		if (parts.length != 6)
			throw new IllegalArgumentException("Invalid FEN");
		
		board.clear();
		
		String[] ranks = parts[0].split("/");
		
		if (ranks.length != 8)
			throw new IllegalArgumentException("Invalid board layout");
		
		for (int rank = 7; rank >= 0; rank--) {
			String row = ranks[7 - rank];
			int file = 0;
			
			for (char c : row.toCharArray()) {
				if (Character.isDigit(c))
					file += c - '0';
				else {
					Space piece = pieceFromFen(c, file, rank);
					
					board.setSpace(file, rank, piece);
					file++;
				}
			}
		}
		
		board.setWhiteToMove(parts[1].equals("w"));
		setCastlingRights(board, parts[2]);
		setEnPassantSquare(board, parts[3]);
		board.setHalfMoveClock(Integer.parseInt(parts[4]));
		board.setFullMoveNumber(Integer.parseInt(parts[5]));
	}
	
	private static char pieceToFen(Space piece) {
		char c = switch (piece.getType()) {
			case KING -> 'k';
			case QUEEN -> 'q';
			case ROOK -> 'r';
			case BISHOP -> 'b';
			case KNIGHT -> 'n';
			case PAWN -> 'p';
			default -> throw new IllegalArgumentException("Unknown piece type.");
		};
		
		return piece.isWhite() ? Character.toUpperCase(c) : c;
	}
	
	private static Space pieceFromFen(char c, int file, int rank) {
		Color white = Character.isUpperCase(c) ? Color.WHITE : Color.BLACK;
		PieceType type = switch (Character.toLowerCase(c)) {
			case 'k' -> PieceType.KING;
			case 'q' -> PieceType.QUEEN;
			case 'r' -> PieceType.ROOK;
			case 'b' -> PieceType.BISHOP;
			case 'n' -> PieceType.KNIGHT;
			case 'p' -> PieceType.PAWN;
			default -> throw new IllegalArgumentException("Invalid FEN piece: " + c);
		};
		
		return new Space(type, white, file, rank);
	}
	
	private static String getCastlingRights(Board board) {
		StringBuilder sb = new StringBuilder();
		
		if (board.isWhiteCanCastleKingside())
			sb.append('K');
		if (board.isWhiteCanCastleQueenside())
			sb.append('Q');
		if (board.isBlackCanCastleKingside())
			sb.append('k');
		if (board.isBlackCanCastleQueenside())
			sb.append('q');
		
		return sb.toString();
	}
	
	private static void setCastlingRights(Board board, String rights) {
		board.setWhiteCanCastleKingside(false);
		board.setWhiteCanCastleQueenside(false);
		board.setBlackCanCastleKingside(false);
		board.setBlackCanCastleQueenside(false);
		
		if (rights.equals("-"))
			return;
		
		for (char c : rights.toCharArray()) {
			switch (c) {
				case 'K' -> board.setWhiteCanCastleKingside(true);
				case 'Q' -> board.setWhiteCanCastleQueenside(true);
				case 'k' -> board.setBlackCanCastleKingside(true);
				case 'q' -> board.setBlackCanCastleQueenside(true);
				default -> throw new IllegalArgumentException("Invalid castling flag: " + c);
			}
		}
	}
	
	private static String getEnPassantSquare(Board board) {
		int enPassantTarget = board.getEnPassantTarget(),
				file = fileOf(enPassantTarget),
				rank = rankOf(enPassantTarget);
		
		if (file == -1 || rank == -1)
			return null;
		
		return "" + (char)('a' + file) + (rank + 1);
	}
	
	private static void setEnPassantSquare(Board board, String square) {
		if (square.equals("-")) {
			board.setEnPassantTarget(-1);
			return;
		}
		
		if (square.length() != 2)
			throw new IllegalArgumentException("Invalid en passant square.");
		
		int file = square.charAt(0) - 'a', rank = square.charAt(1) - '1';
		
		if (file < 0 || file > 7 || rank < 0 || rank > 7)
			throw new IllegalArgumentException("Invalid en passant square.");
		
		board.setEnPassantTarget(indexOf(file, rank));
	}
}
