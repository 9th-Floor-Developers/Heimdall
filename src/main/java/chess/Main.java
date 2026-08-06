package chess;

import chess.model.Color;
import chess.players.*;

import java.util.HashMap;

public class Main {
	public static void main(String[] args) {
		HashMap<Color, Player> players = new HashMap<>();
		players.put(Color.BLACK, new BruteForceBot(3, 6));
//		players.put(Color.BLACK, new UiPlayer());
		players.put(Color.WHITE, new RandomBot());

		GameManager.runGame(players);
	}
}
