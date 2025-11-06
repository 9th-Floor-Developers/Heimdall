package shapes;

public class Shape {
	public ShapeType type;
	public float[][] image;
	
	public Shape(ShapeType type, float[][] image) {
		this.type = type;
		this.image = image;
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (float[] floats : image) {
			for (float f : floats)
				builder.append((int) f);
			builder.append("\n");
		}
		
		return builder.toString();
	}
	
	public float[] to1D() {
		float[] line = new float[(int) Math.pow(image.length, 2)];
		
		int i = 0;
		for (float[] floats : image) {
			for (float f : floats) {
				line[i] = f;
				i++;
			}
		}
		
		return line;
	}
}
