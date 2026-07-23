package chess;
import static chess.ChessUtils.printBoard;
import chess.model.Color;
import chess.model.Move;

import java.util.ArrayList;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Board board = new Board();
		Scanner scanner = new Scanner(System.in);
		
		while (true) {
			printBoard(board);
			ArrayList<Move> legalMoves = MoveGenerator.generateLegalMoves(board);
			
			if (legalMoves.isEmpty()) {
				if (board.isInCheck(board.isWhiteToMove() ? Color.WHITE : Color.BLACK))
					System.out.println("Checkmate! " + (board.isWhiteToMove() ? "Black" : "White") + " wins.");
				else
					System.out.println("Stalemate! Draw.");
				break;
			}
			
			if (board.getHalfMoveClock() >= 100) {
				System.out.println("Draw by 50-move rule.");
				break;
			}
			
			System.out.print("Enter move (e.g. e2e4, e7e8q), or 'moves' to list legal moves, 'quit' to exit: ");
			String input = scanner.nextLine().trim();
			
			if (input.equalsIgnoreCase("quit"))
				break;
			
			if (input.equalsIgnoreCase("moves")) {
				for (Move m : legalMoves)
					System.out.print(m.toLongAlgebraic() + " ");
				System.out.println();
				continue;
			}
			
			Move chosen = parseMove(input, legalMoves);
			if (chosen == null) {
				System.out.println("Invalid or illegal move: " + input);
				continue;
			}
			board.makeMove(chosen);
		}
	}
	
	private static Move parseMove(String input, ArrayList<Move> legalMoves) {
		for (Move m : legalMoves)
			if (m.toLongAlgebraic().equalsIgnoreCase(input))
				return m;
		return null;
	}
}
