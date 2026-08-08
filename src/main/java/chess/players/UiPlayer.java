package chess.players;

import chess.Board;
import chess.model.Color;
import chess.model.Move;

import java.util.HashSet;
import java.util.concurrent.SynchronousQueue;

public class UiPlayer implements Player {
	private final SynchronousQueue<Move> moveExchange = new SynchronousQueue<>();
	private HashSet<Move> pendingLegalMoves;
	
	@Override
	public String getDisplayName() {
		return "UI Player";
	}
	
	@Override
	public Move getNextMove(HashSet<Move> legalMoves, Board board, Color color) {
		pendingLegalMoves = legalMoves;
		try {
			return moveExchange.take();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException("Interrupted waiting for UI move", e);
		}
	}
	
	public void submitMove(Move move) {
		if (pendingLegalMoves == null || !pendingLegalMoves.contains(move))
			return;
		try {
			moveExchange.put(move);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}
}
