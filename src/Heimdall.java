import model.data.NumberImage;
import static utils.NumberUtils.getRandomImgs;

public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		NumberImage[] images = getRandomImgs("./src/datasets/numbers/", 1000);
		float[][] targets = new float[images.length][],
				inputs = new float[images.length][];
		int[] outputs = new int[images.length];
		
		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i].scaleDownImage(5);
			inputs[i] = image.to1D();
			targets[i] = image.toTarget();
			outputs[i] = image.value();
		}
		
		Trainer trainer = new Trainer(
				10,  // number of agents per round, more possibilities to evolve
				new int[] {  // layers format
						inputs[0].length,  // input layer must match input count // number of middle layer nodes, more opportunities per agent to learn
						100,
						targets[0].length  // output layer is number of possible answers (0.0-1.0 inclusive)
				}
		);
		
		for (int generation = 1; generation <= 100000; generation++) {
			trainer.train(
					inputs,
					targets,
					outputs,
					.01f,
					generation
			);
		}
		
		System.out.println("Best Score: " + trainer.getBestScore());
		
		trainer.logWeights();
		trainer.logBiases();
	}
}
