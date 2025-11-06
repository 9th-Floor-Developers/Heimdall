package numbers;

/**
 * Record class representing an image of a number from the dataset.
 *
 * @param pixels 2D float array representing all pixel values in greyscale
 * @param value actual value of the image, assigned dynamically, not guessed
 */
public record NumberImage(float[][] pixels,
                          int value) {
	
	/**
	 * Prints the image in ASCII format using pixels variable.
	 * <p>
	 * Greyscale values are printed as @$#!.? in .2 intervals.
	 */
	public void printASCII() {
		for (float[] floats : pixels) {
			for (float v : floats) {
				char symbol = '?';
				
				if (v >= .8f)
					symbol = '@';
				else if (v < .8f && v >= .6f)
					symbol = '$';
				else if (v < .6f && v >= .4f)
					symbol = '#';
				else if (v < .4f && v >= .2f)
					symbol = '!';
				else if (v < .2f)
					symbol = '.';
				
				System.out.print(symbol);
			}
			
			System.out.println();
		}
	}
}
