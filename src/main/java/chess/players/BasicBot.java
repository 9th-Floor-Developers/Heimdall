package chess.players;

import chess.Board;
import chess.model.Color;
import chess.model.Move;

import java.util.ArrayList;

public class BasicBot implements Player{
    @Override
    public String getDisplayName() {
        return "Basic boi";
    }

    @Override
    public Move getNextMove(ArrayList<Move> legalMoves, Board board, Color color) {
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
