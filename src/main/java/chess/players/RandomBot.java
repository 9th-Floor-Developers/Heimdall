package chess.players;

import chess.Board;
import chess.model.Color;
import chess.model.Move;

import java.util.HashSet;
import java.util.Random;

public class RandomBot implements Player{
    @Override
    public String getDisplayName() {
        return "Random Bot";
    }

    @Override
    public Move getNextMove(HashSet<Move> legalMoves, Board board, Color color) {
        Random random = new Random();
		return legalMoves.toArray(new Move[0])[random.nextInt(legalMoves.size())];
    }
}
