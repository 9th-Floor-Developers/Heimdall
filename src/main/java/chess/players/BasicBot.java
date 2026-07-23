package chess.players;

import chess.Board;
import chess.model.Move;
import chess.model.PieceType;
import static chess.model.PieceType.*;

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

    public int evalBoard(Board board) {
        int score = 0;
	    
	    score += board.getPieceTypeAmount(PAWN);
	    score += board.getPieceTypeAmount(KNIGHT) * KNIGHT.getMaterial();
	    score += board.getPieceTypeAmount(BISHOP) * BISHOP.getMaterial();
	    score += board.getPieceTypeAmount(ROOK) * ROOK.getMaterial();
	    score += board.getPieceTypeAmount(QUEEN) * QUEEN.getMaterial();
		
		// TODO: possibly implement relative value

        return score;
    }
}
