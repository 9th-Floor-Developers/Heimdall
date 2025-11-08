import static numbers.NumberGenerator.getAllImgs;
import numbers.NumberImage;

import java.util.Random;

public class Heimdall {
	public static void main(String[] args) throws Exception {
		NumberImage[] images = getAllImgs();
		
		float[][] allInputs = new float[images.length][];
		int[] allOutputs = new int[images.length];
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
			
			allInputs[i] = flatInputs;
			allOutputs[i] = image.value();
		}

        int data_amount = 100;

        float[][] inputs = new float[data_amount][];
        int[] outputs = new int[data_amount];
        for (int i = 0; i < data_amount; i++) {
            Random random = new Random();
            int idx = random.nextInt(0, allInputs.length);
            inputs[i] = allInputs[idx];
            outputs[i] = allOutputs[idx];
        }
		
		Trainer trainer = new Trainer(
				10,  // number of agents per round, more possibilities to evolve
				new int[] {  // layers format
						inputs[0].length,  // input layer must match input count
						15, // number of middle layer nodes, more opportunities per agent to learn
						11  // output layer is number of possible answers (0.0-1.0 inclusive)
				}
		);
		trainer.train(
				inputs,
				outputs,
				100, // number of rounds, more opportunities to get higher percentage
				.5f
		);
	}
}
