package chess.players;

import chess.Board;
import chess.MoveGenerator;
import chess.model.Color;
import chess.model.Move;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static chess.model.PieceType.KING;

public class BruteForceBot implements Player{
    public int baseDepth;
    public int endgameDepth;

    public BruteForceBot(int baseDepth, int endgameDepth) {
        this.baseDepth = baseDepth;
        this.endgameDepth = endgameDepth;
    }

    @Override
    public String getDisplayName() {
        return "Brute forcer";
    }

    @Override
    public Move getNextMove(ArrayList<Move> legalMoves, Board board) {
        int depth = board.isEndgame() ? endgameDepth : baseDepth;

        AtomicReference<Move> bestMove = new AtomicReference<>();
        AtomicInteger bestScore = new  AtomicInteger(Integer.MIN_VALUE);
        ArrayList<Thread> threads = new ArrayList<>();
        Object lock = new Object();
        int inf = Integer.MAX_VALUE;

        for (Move move : legalMoves) {
            Board child = board.clone();
            child.makeMove(move);

            Thread thread = new Thread(() -> {
                int score = -search(child, depth - 1, -inf, inf);

                synchronized (lock) {
                    if (score > bestScore.get()) {
                        bestScore.set(score);
                        bestMove.set(move);
                    }
                }
            });

            thread.start();
            threads.add(thread);
        }

        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        return bestMove.get();
    }

    private static int search(Board board, int depth, int alpha, int beta) {
        ArrayList<Move> moves = MoveGenerator.generateLegalMoves(board);
        if (moves.isEmpty())
            return (board.isInCheck(board.isWhiteToMove() ? Color.WHITE : Color.BLACK))
                    ? -KING.getMaterial() - depth  // checkmate
                    : 0;  // stalemate

        if (depth == 0)
            return board.evalBoard();

        int best = Integer.MIN_VALUE;

        for (Move move : moves) {
            Board clone = board.clone();
            clone.makeMove(move);

            int score = -search(clone, depth - 1, -beta, -alpha);
            best = Math.max(best, score);
            alpha = Math.max(alpha, score);

            if (alpha >= beta)
                break;
        }

        return best;
    }
}
