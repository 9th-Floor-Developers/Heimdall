package chess.players;

import chess.model.Move;

import java.util.ArrayList;

public interface Player {
    String getDisplayName();

    Move getNextMove(ArrayList<Move> legalMoves);
}
