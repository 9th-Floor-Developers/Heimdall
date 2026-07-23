package chess.players;

import chess.model.Move;

import java.util.ArrayList;
import java.util.Random;

public class RandomBot implements Player{
    @Override
    public String getDisplayName() {
        return "Random Bot";
    }

    @Override
    public Move getNextMove(ArrayList<Move> legalMoves) {
        Random random = new Random();
        return legalMoves.get(random.nextInt(legalMoves.size()));
    }
}
