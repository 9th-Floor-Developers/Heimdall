package chess.ui;

import chess.Board;

import javax.swing.*;
import java.awt.*;

public final class MainWindow extends JFrame {
	private JPanel rootPanel;
	private JPanel boardPanel;
	private JButton saveButton;
	private JButton importButton;
	private JButton undoButton;
	private JButton quitButton;
	
	public MainWindow(Board game) {
		setTitle("Heimdall | Chess");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		setVisible(true);
		setBackground(Color.BLUE);
		pack();
//		setResizable(false);
		setContentPane(rootPanel);
		
		((BoardPanel) boardPanel).setGame(game);
		
		quitButton.addActionListener(e -> System.exit(0));
	}
	
	private void createUIComponents() {
		boardPanel = new BoardPanel();
	}
}
