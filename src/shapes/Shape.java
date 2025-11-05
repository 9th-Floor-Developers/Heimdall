package shapes;

public class Shape {
    public enum ShapeType{
        Circle,
        Square,
        Triangle,
    }
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
            for (int x = 0; x < image.length; x++) {
                builder.append((int) floats[x]);
            }
            builder.append("\n");
        }

        return builder.toString();
    }

    public float[] to1D(){
        float[] line = new float[(int) Math.pow(image.length, 2)];

        int i = 0;
        for (float[] floats : image) {
            for (int x = 0; x < image.length; x++) {
                line[i] = floats[x];
                i++;
            }
        }

        return line;
    }
}
