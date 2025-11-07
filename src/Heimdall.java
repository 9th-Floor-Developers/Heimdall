import static numbers.NumberGenerator.getAllImgs;
import numbers.NumberImage;

public class Heimdall {
	public static void main(String[] args) throws Exception {
		BasicTrainer trainer = new BasicTrainer();
		NumberImage[] images = getAllImgs();
		
		float[][] inputs = new float[images.length][];
		int[] outputs = new int[images.length];
		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i];
			float[] flatInputs = new float[image.pixels().length * image.pixels()[0].length];
			int idx = 0;
			for (int r = 0; r < image.pixels().length; r++) {
				for (int c = 0; c < image.pixels()[0].length; c++) {
					flatInputs[idx] = image.pixels()[r][c];
					idx++;
				}
			}
			
			inputs[i] = flatInputs;
			outputs[i] = image.value();
		}
		
		trainer.train(
				inputs,
				outputs,
				new int[] {  // layers format
						inputs[0].length,  // input layer must match input count
						15, // number of middle layer nodes, more opportunities per agent to learn
						15,
						15,
						11  // output layer is number of possible answers (0.0-1.0 inclusive)
				},
				10,  // number of agents per round, more possibilities to evolve
				999,// number of rounds, more opportunities to get higher percentage
                0.5f
		);
	}
}
