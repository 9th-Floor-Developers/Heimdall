package numbers;

public record NumberImage(float[][] pixels,
                          int value) {
	
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
