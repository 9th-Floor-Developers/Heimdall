package chess;

import static chess.ChessUtils.printBoard;
import chess.model.Color;
import chess.model.Move;

import chess.players.Player;

import java.util.HashMap;
import java.util.HashSet;

public class GameManager {
	public static GameManager instance;
	
	public GameManager() {
		instance = this;
	}

	public static Color runGame(HashMap<Color, Player> players) {
		Board board = new Board();
		
		System.out.println("Starting game between " + players.get(Color.WHITE).getDisplayName() + " and " + players.get(Color.BLACK).getDisplayName());
		
		while (true) {
			printBoard(board);
			HashSet<Move> legalMoves = MoveGenerator.generateLegalMoves(board);
			
			if (legalMoves.isEmpty()) {
				if (board.isInCheck(board.getTurnColor())) {
					System.out.println("Checkmate! " + (board.getOppositeColor().toString()) + " wins.");
					return board.getOppositeColor();
				}
				else {
					System.out.println("Stalemate! Draw.");
					return Color.NONE;
				}
			}
			
			if (board.getHalfMoveClock() >= 100) {
				System.out.println("Draw by 50-move rule.");
				return Color.NONE;
			}
			
//			System.out.println("Best Move: " + findBestMove(board, 4).toLongAlgebraic());

			Move chosen = players.get(board.getTurnColor()).getNextMove(legalMoves, board, board.getTurnColor());

			if (chosen == null){
				System.out.println((board.getTurnColor().toString() + " Forfeited"));
				return board.getOppositeColor();//The other side that did not forfeit wins
			}
			
			board.makeMove(chosen);
			
			System.out.println("==============================================================");
		}
	}
}
