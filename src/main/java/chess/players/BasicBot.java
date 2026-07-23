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

    public int evalBoard(Board board) {
        int score = 0;
		
		HashMap<PieceType, Integer> pieces = board.getPieces();
	    
        score += pieces.get(PAWN);
        score += pieces.get(KNIGHT) * KNIGHT.getMaterial();
        score += pieces.get(BISHOP) * BISHOP.getMaterial();
        score += pieces.get(ROOK) * ROOK.getMaterial();
        score += pieces.get(QUEEN) * QUEEN.getMaterial();
		
		// TODO: possibly implement relative value

        return score;
    }
}
