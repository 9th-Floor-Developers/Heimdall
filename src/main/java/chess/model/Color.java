package chess.model;

public enum Color {
	BLACK,
	WHITE,
	NONE;
	
	public Color getOpposite() {
		return switch (this) {
			case BLACK -> WHITE;
			case WHITE -> BLACK;
			case NONE -> throw new RuntimeException("No Opposite Color");
		};
	}
}
