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
