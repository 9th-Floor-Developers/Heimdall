package chess.players;

import chess.Board;
import chess.model.Color;
import chess.model.Move;

import java.util.ArrayList;

public interface Player {
    String getDisplayName();

    Move getNextMove(ArrayList<Move> legalMoves, Board board, Color color);
}
