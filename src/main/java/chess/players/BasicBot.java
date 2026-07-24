package chess.players;

import chess.Board;
import chess.model.Move;
import chess.model.PieceType;
import static chess.model.PieceType.*;

import java.util.ArrayList;
import java.util.HashMap;

public class BasicBot implements Player{
    @Override
    public String getDisplayName() {
        return "Basic boi";
    }

    @Override
    public Move getNextMove(ArrayList<Move> legalMoves, Board board) {
        /*
        double bestScore = Double.MIN_NORMAL;
        Move bestMove = legalMoves.getFirst();

        for (Move legalMove : legalMoves){
            Board newBoard = board.clone();
            newBoard.makeMove(legalMove);
            double score = evalBoard(newBoard);

            if (score > bestScore){
                bestMove = legalMove;
                bestScore = score;
            }

        }

        return bestMove;
         */
        return legalMoves.getFirst();
    }
}
