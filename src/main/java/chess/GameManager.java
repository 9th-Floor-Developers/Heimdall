package chess;

import static chess.ChessUtils.printBoard;
import chess.model.Color;
import chess.model.Move;
import static chess.model.PieceType.KING;
import chess.players.Player;
import chess.players.RandomBot;
import chess.players.TerminalPlayer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class GameManager {
	public static GameManager i;
	
	public GameManager() {
		i = this;
	}
	
	public static void run() {
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
			
//			System.out.println("Best Move: " + findBestMove(board, 4).toLongAlgebraic());

//			Move chosen = board.isWhiteToMove() ? players.get(Color.WHITE).getNextMove(legalMoves, board) : players.get(Color.BLACK).getNextMove(legalMoves, board);
			int depth = (board.isEndgame()) ? 6 : 3;
			Move chosen = board.isWhiteToMove() ? findBestMove(board, depth) : players.get(Color.BLACK).getNextMove(legalMoves, board);
			if (chosen == null){
				System.out.println((board.isWhiteToMove() ? "Black" : "White") + " Forfeited");
				break;
			}
			
			board.makeMove(chosen);
			
			System.out.println("==============================================================");
		}
	}
	
	private static Move findBestMove(Board board, int depth) {
		ArrayList<Move> moves = MoveGenerator.generateLegalMoves(board);
		if (moves.isEmpty())
			return null;
		
		AtomicReference<Move> bestMove = new AtomicReference<>();
		AtomicInteger bestScore = new  AtomicInteger(Integer.MIN_VALUE);
		ArrayList<Thread> threads = new ArrayList<>();
		Object lock = new Object();
		int inf = Integer.MAX_VALUE;
		
		for (Move move : moves) {
			Board child = board.clone();
			child.makeMove(move);
			
			Thread thread = new Thread(() -> {
				int score = -search(child, depth - 1, -inf, inf);
				
				synchronized (lock) {
					if (score > bestScore.get()) {
						bestScore.set(score);
						bestMove.set(move);
					}
				}
			});
			
			thread.start();
			threads.add(thread);
		}
		
		for (Thread thread : threads) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}
		}
		
		return bestMove.get();
	}
	
	private static int search(Board board, int depth, int alpha, int beta) {
		ArrayList<Move> moves = MoveGenerator.generateLegalMoves(board);
		if (moves.isEmpty())
			return (board.isInCheck(board.isWhiteToMove() ? Color.WHITE : Color.BLACK))
					? -KING.getMaterial() - depth  // checkmate
					: 0;  // stalemate
		
		if (depth == 0)
			return board.evalBoard();
		
		int best = Integer.MIN_VALUE;
		
		for (Move move : moves) {
			Board clone = board.clone();
			clone.makeMove(move);
			
			int score = -search(clone, depth - 1, -beta, -alpha);
			best = Math.max(best, score);
			alpha = Math.max(alpha, score);
			
			if (alpha >= beta)
				break;
		}
		
		return best;
	}
}
