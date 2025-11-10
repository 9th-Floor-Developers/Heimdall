import model.data.NumberImage;
import static utils.NumberUtils.getAllImgs;

public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		NumberImage[] images = getAllImgs("./src/datasets/numbers/");
		
		for (int i = 0; i < images.length; i++)
			images[i] = images[i].scaleDownImage(5);  // scale images down for speed and accuracy
		
		float[][] inputs = new float[images.length][];
		int[] outputs = new int[images.length];
		
		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i];
			inputs[i] = image.to1D();
			outputs[i] = image.value();
		}

//		float[][] inputs = or5_inputs;
//		int[] outputs = or5_outputs;
		
		Trainer trainer = new Trainer(
				4,  // number of agents per round, more possibilities to evolve
				new int[] {  // layers format
						inputs[0].length,  // input layer must match input count
						200, // number of middle layer nodes, more opportunities per agent to learn
						10  // output layer is number of possible answers (0.0-1.0 inclusive)
				}
		);
		trainer.train(
				inputs,
				outputs,
				50, // number of rounds, more opportunities to get higher percentage
				.05f
		);
	}
}
