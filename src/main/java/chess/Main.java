package chess;

import chess.model.Color;
import chess.players.BruteForceBot;
import chess.players.Player;
import chess.players.RandomBot;
import chess.ui.MainWindow;

import java.util.HashMap;

public class Main {
	public static void main(String[] args) {
		HashMap<Color, Player> players = new HashMap<>();
		players.put(Color.BLACK, new BruteForceBot(3, 6));
		players.put(Color.WHITE, new RandomBot());


		GameManager.runGame(players);
//		MainWindow window = new MainWindow();
	}
}
