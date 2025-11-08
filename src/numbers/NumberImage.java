package numbers;

/**
 * Record class representing an image of a number from the dataset.
 *
 * @param pixels 2D float array representing all pixel values in greyscale
 * @param value actual value of the image, assigned dynamically, not guessed
 */
public record NumberImage(float[][] pixels,
                          int value) {

    public float[] to1D() {
        float[] flatInputs = new float[pixels().length * pixels()[0].length];
        int idx = 0;
        for (int r = 0; r < pixels().length; r++) {
            for (int c = 0; c < pixels()[0].length; c++) {
                flatInputs[idx] = pixels()[r][c];
                idx++;
            }
        }

        return flatInputs;
    }

    public float[] toTarget(){
        float[] target = new float[10];
        target[value()] = 1;
        return target;
    }

	/**
	 * Prints the image in ASCII format using pixels variable.
	 * <p>
	 * Greyscale values are printed as @$#!.? in .2 intervals.
	 */
	public void printASCII() {
		for (float[] floats : pixels) {
			for (float v : floats) {
				char symbol = '?';
				
				if (v >= .9f)
					symbol = '@';
				else if (v >= .8f)
					symbol = '$';
				else if (v >= .7f)
					symbol = '#';
				else if (v >= .6f)
					symbol = '!';
                else if (v >= .5f)
                    symbol = ';';
                else if (v >= .4f)
                    symbol = ':';
                else if (v >= .3f)
                    symbol = '~';
                else if (v >= .2f)
                    symbol = '-';
                else if (v >= .1f)
                    symbol = ',';
				else
					symbol = '.';
				
				System.out.print(symbol);
			}
			
			System.out.println();
		}
	}
}
