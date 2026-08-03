package chess;

import static chess.ChessUtils.printBoard;
import chess.model.Color;
import chess.model.Move;

import chess.players.Player;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class GameManager {
	public static GameManager instance;
	
	public GameManager() {
		instance = this;
	}

	public static void runGameBatch(HashMap<Color, Player> players, int amount){
		List<Color> winners = new ArrayList<>();

		for (int i = 0; i < amount; i++){
			winners.add(runGame(players));
		}

		System.out.println("============================================");
		System.out.println("Final tally");
		System.out.println("White: " + winners.stream().filter(n -> n.equals(Color.WHITE)).count());
		System.out.println("Black: " + winners.stream().filter(n -> n.equals(Color.BLACK)).count());
		System.out.println("Draw: " + winners.stream().filter(n -> n.equals(Color.NONE)).count());
	}

	public static Color runGame(HashMap<Color, Player> players) {
		Board board = new Board();
		
		System.out.println("Starting game between " + players.get(Color.WHITE).getDisplayName() + " and " + players.get(Color.BLACK).getDisplayName());
		
		while (true) {
			printBoard(board);
			window.setStatusLabel(board.toMoveColor() + " To Move");
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
