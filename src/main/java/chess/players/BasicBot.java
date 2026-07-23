package chess.players;

import chess.Board;
import chess.model.Move;
import chess.model.PieceType;

import java.util.ArrayList;
import java.util.Comparator;

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

    public double evalBoard(Board board){
        double score = 0;

        score += board.getPieceTypeAmount(PieceType.PAWN) * 1.0;
        score += board.getPieceTypeAmount(PieceType.KNIGHT) * 3.0;
        score += board.getPieceTypeAmount(PieceType.BISHOP) * 3.5;
        score += board.getPieceTypeAmount(PieceType.ROOK) * 5.0;
        score += board.getPieceTypeAmount(PieceType.QUEEN) * 10.0;
        score += board.getPieceTypeAmount(PieceType.KING) * 100.0;

        return score;
    }
}
