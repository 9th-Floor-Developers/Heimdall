package chess.ui;

import chess.Board;

import javax.swing.*;
import java.awt.*;

public class MainWindow extends JFrame {
	private JPanel rootPanel;
	private JPanel boardPanel;
	private JButton saveButton;
	private JButton importButton;
	private JButton undoButton;
	private JButton quitButton;
	
	private Board game;
	
	public MainWindow(Board game) {
		setTitle("Heimdall | Chess");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		setVisible(true);
		setBackground(Color.BLUE);
		pack();
//		setResizable(false);
		setContentPane(rootPanel);
		
		quitButton.addActionListener(e -> System.exit(0));
	}
	
	private void createUIComponents() {
		boardPanel = new BoardPanel(game);
	}
}
