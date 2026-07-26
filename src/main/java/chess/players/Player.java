package chess.players;

import chess.Board;
import chess.model.Color;
import chess.model.Move;

import java.util.HashSet;

public interface Player {
    String getDisplayName();

    Move getNextMove(HashSet<Move> legalMoves, Board board, Color color);
}
