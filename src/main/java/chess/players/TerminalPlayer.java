package chess.players;

import chess.Board;
import chess.model.Color;
import chess.model.Move;

import java.util.HashSet;
import java.util.Scanner;


public class TerminalPlayer implements Player {
    @Override
    public String getDisplayName() {
        return "Terminal Player";
    }

    @Override
    public Move getNextMove(HashSet<Move> legalMoves, Board board, Color color) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter move (e.g. e2e4, e7e8q), or 'moves' to list legal moves, 'quit' to exit: ");
        String input = scanner.nextLine().trim();

        if (input.equalsIgnoreCase("quit"))
            return null;

        if (input.equalsIgnoreCase("moves")) {
            for (Move m : legalMoves)
                System.out.print(m.toLongAlgebraic() + " ");
            System.out.println();
            return getNextMove(legalMoves, board, color);
        }

        Move chosen = parseMove(input, legalMoves);
        if (chosen == null) {
            System.out.println("Invalid or illegal move: " + input);
            System.out.println("Please try again");
            return getNextMove(legalMoves, board, color);
        }
        return chosen;
    }

    private static Move parseMove(String input, HashSet<Move> legalMoves) {
        for (Move m : legalMoves)
            if (m.toLongAlgebraic().equalsIgnoreCase(input))
                return m;
        return null;
    }
}
