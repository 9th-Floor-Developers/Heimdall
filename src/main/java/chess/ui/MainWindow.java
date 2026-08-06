package chess.ui;

import chess.Board;
import chess.FenUtils;
import chess.players.UiPlayer;

import javax.swing.*;

public final class MainWindow extends JFrame {
	private JPanel rootPanel;
	private JPanel boardPanel;
	private JButton saveButton;
	private JButton importButton;
	private JButton undoButton;
	private JButton quitButton;
	private JLabel statusLabel;
	
	public MainWindow(Board game) {
		setTitle("Heimdall | Chess");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
		setVisible(true);
		pack();
//		setResizable(false);
		setContentPane(rootPanel);
		
		((BoardPanel) boardPanel).setGame(game);
		
		undoButton.addActionListener(e -> {
			game.undoMove();
			boardPanel.repaint();
		});
		saveButton.addActionListener(e -> {
			JOptionPane.showMessageDialog(
					this,
					FenUtils.exportFen(game)
			);
		});
		importButton.addActionListener(e -> {
			String fenString = JOptionPane.showInputDialog(
					this,
					"Enter FEN String"
			);
			if (fenString != null && !fenString.trim().isEmpty()) {
				FenUtils.importFen(game, fenString);
				boardPanel.repaint();
			}
		});
		quitButton.addActionListener(e -> System.exit(0));
	}
	
	private void createUIComponents() {
		boardPanel = new BoardPanel();
	}
	
	public void setStatusLabel(String text) {
		statusLabel.setText("Status: " + text);
	}
	
	public void setBoardPanelUiPlayer(UiPlayer player) {
		((BoardPanel) boardPanel).setUiPlayer(player);
	}
}
