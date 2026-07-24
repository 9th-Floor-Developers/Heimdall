package chess;

import static chess.model.Color.WHITE;
import chess.model.Space;

public class ChessUtils {
	public static final int[][] DIAG_DIRS = {
			{ 1, -1 },      { 1, 1 },
			
			{ -1, -1 },     { -1, 1 }
	};
	
	public static final int[][] ORTHO_DIRS = {
						{ 1, 0 },
			{ -1, 0 },              { 0, 1 },
						{ 0, -1 }
	};
	
	public static final int[][] ALL_DIRS = {
			{ -1, 1 },  { 0, 1 },   { 1, 1 },
			{ -1, 0 },              { 1, 0 },
			{ -1, -1 }, { 0, -1 },  { 1, -1 }
	};
	
	public static final int[][] KNIGHT_DIRS = {
						{ -1, 2 },      { 1, 2 },
			{ -2, 1 },                              { 2, 1 },
			
			{ -2, -1 },                             { 2, -1 },
						{ -1, -2 },     { 1, -2 },
	};
	
	public static int[] PAWN_TABLE = {
			0,  0,  0,  0,  0,  0,  0,  0,
			50, 50, 50, 50, 50, 50, 50, 50,
			10, 10, 20, 30, 30, 20, 10, 10,
			5,  5, 10, 27, 27, 10,  5,  5,
			0,  0,  0, 25, 25,  0,  0,  0,
			5, -5,-10,  0,  0,-10, -5,  5,
			5, 10, 10,-25,-25, 10, 10,  5,
			0,  0,  0,  0,  0,  0,  0,  0
	};
	
	public static int[] KNIGHT_TABLE = {
			-50,-40,-30,-30,-30,-30,-40,-50,
			-40,-20,  0,  0,  0,  0,-20,-40,
			-30,  0, 10, 15, 15, 10,  0,-30,
			-30,  5, 15, 20, 20, 15,  5,-30,
			-30,  0, 15, 20, 20, 15,  0,-30,
			-30,  5, 10, 15, 15, 10,  5,-30,
			-40,-20,  0,  5,  5,  0,-20,-40,
			-50,-40,-20,-30,-30,-20,-40,-50,
	};
	
	public static int[] BISHOP_TABLE = {
			-20,-10,-10,-10,-10,-10,-10,-20,
			-10,  0,  0,  0,  0,  0,  0,-10,
			-10,  0,  5, 10, 10,  5,  0,-10,
			-10,  5,  5, 10, 10,  5,  5,-10,
			-10,  0, 10, 10, 10, 10,  0,-10,
			-10, 10, 10, 10, 10, 10, 10,-10,
			-10,  5,  0,  0,  0,  0,  5,-10,
			-20,-10,-40,-10,-10,-40,-10,-20,
	};
	
	public static final int[] ROOK_TABLE = {
			0,  0,  0,  5,  5,  0,  0,  0,
			-5,  0,  0,  0,  0,  0,  0, -5,
			-5,  0,  0,  0,  0,  0,  0, -5,
			-5,  0,  0,  0,  0,  0,  0, -5,
			-5,  0,  0,  0,  0,  0,  0, -5,
			-5,  0,  0,  0,  0,  0,  0, -5,
			5, 10, 10, 10, 10, 10, 10,  5,
			0,  0,  0,  0,  0,  0,  0,  0
	};
	
	public static final int[] QUEEN_TABLE = {
			-20,-10,-10, -5, -5,-10,-10,-20,
			-10,  0,  0,  0,  0,  0,  0,-10,
			-10,  0,  5,  5,  5,  5,  0,-10,
			 -5,  0,  5,  5,  5,  5,  0, -5,
			  0,  0,  5,  5,  5,  5,  0, -5,
			-10,  5,  5,  5,  5,  5,  0,-10,
			-10,  0,  5,  0,  0,  0,  0,-10,
			-20,-10,-10, -5, -5,-10,-10,-20
	};
	
	
	public static int fileOf(int index) {
		return index % 8;
	}
	
	public static int rankOf(int index) {
		return index / 8;
	}
	
	public static int indexOf(int file, int rank) {
		return rank * 8 + file;
	}
	
	public static boolean onBoard(int file, int rank) {
		return file >= 0 && file < 8 && rank >= 0 && rank < 8;
	}
	
	public static String squareName(Space space) {
		return "" + (char) ('a' + space.getFile()) + (space.getRank() + 1);
	}
	
	public static void printBoard(Board board) {
		for (int rank = 7; rank >= 0; rank--) {
			StringBuilder sb = new StringBuilder();
			sb.append(rank + 1).append(" ");
			
			for (int file = 0; file < 8; file++) {
				Space space = board.pieceAt(file, rank);
				char letter = space.getType().getLetter();
				
				if (space.getColor() == WHITE)
					letter = Character.toUpperCase(letter);
				
				sb.append(letter).append(" ");
			}
			
			System.out.println(sb);
		}
		
		System.out.println("  a b c d e f g h");
		System.out.println((board.isWhiteToMove() ? "White" : "Black") + " to move");
	}
}
