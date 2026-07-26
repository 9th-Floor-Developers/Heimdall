package chess;

import static chess.ChessUtils.printBoard;
import chess.model.Color;
import chess.model.Move;

import chess.players.BruteForceBot;
import chess.players.Player;
import chess.players.RandomBot;

import java.util.ArrayList;
import java.util.HashMap;

public class GameManager {
	public static GameManager i;
	
	public GameManager() {
		i = this;
	}
	
	public static void run() {
		Board board = new Board();
		
		HashMap<Color, Player> players = new HashMap<>();
		players.put(Color.BLACK, new BruteForceBot(3, 6));
		players.put(Color.WHITE, new RandomBot());
		
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
			
//			System.out.println("Best Move: " + findBestMove(board, 4).toLongAlgebraic());

			Move chosen = board.isWhiteToMove() ?
					players.get(Color.WHITE).getNextMove(legalMoves, board, Color.WHITE)
					: players.get(Color.BLACK).getNextMove(legalMoves, board, Color.BLACK);
			if (chosen == null){
				System.out.println((board.isWhiteToMove() ? "Black" : "White") + " Forfeited");
				break;
			}
			
			board.makeMove(chosen);
			
			System.out.println("==============================================================");
		}
	}
}
