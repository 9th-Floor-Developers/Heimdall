package shapes;

import java.util.Random;

public class ShapeGenerator {
	public static Shape generateSquare() {
		Random random = new Random();
		
		float[][] image = new float[12][12];
		int startX = random.nextInt(0, 5);
		int startY = random.nextInt(0, 5);
		
		int sizeX = random.nextInt(2, 6);
		int sizeY = random.nextInt(2, 6);
		
		
		for (int x = startX; x <= startX + sizeX; x++)
			image[startY][x] = 1;
		
		for (int x = startX; x <= startX + sizeX; x++)
			image[startY + sizeY][x] = 1;
		
		for (int y = startY; y <= startY + sizeY; y++)
			image[y][startX] = 1;
		
		for (int y = startY; y <= startY + sizeY; y++)
			image[y][startX + sizeX] = 1;
		
		return new Shape(ShapeType.Square, image);
	}
	
	public static Shape generateCircle() {
		Random random = new Random();
		
		float[][] image = new float[12][12];
		int centerX = random.nextInt(0, 5);
		int centerY = random.nextInt(0, 5);
		
		int radius = random.nextInt(2, 4);
		
		for (int y = 0; y < image.length; y++)
			for (int x = 0; x < image.length; x++)
				if (Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2) < Math.pow(radius, 2))
					image[y][x] = 1;
		
		return new Shape(ShapeType.Square, image);
	}
}
