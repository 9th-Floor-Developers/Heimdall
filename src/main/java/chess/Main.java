package chess;
import static chess.ChessUtils.printBoard;
import chess.model.Color;
import chess.model.Move;
import chess.players.Player;
import chess.players.RandomBot;
import chess.players.TerminalPlayer;

import java.util.ArrayList;
import java.util.HashMap;

public class Main {
	public static void main(String[] args) {
		Board board = new Board();

		HashMap<Color, Player> players = new HashMap<>();
		players.put(Color.WHITE, new TerminalPlayer());
		players.put(Color.BLACK, new RandomBot());

		System.out.println("Starting game between " + players.get(Color.WHITE).getDisplayName() + " and " + players.get(Color.BLACK).getDisplayName());

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

			Move chosen = board.isWhiteToMove() ? players.get(Color.WHITE).getNextMove(legalMoves) : players.get(Color.BLACK).getNextMove(legalMoves);
			if (chosen == null){
				System.out.println((board.isWhiteToMove() ? "Black" : "White") + "Forfeited");
				break;
			}
			board.makeMove(chosen);
			System.out.println("==============================================================");
		}
	}
}
