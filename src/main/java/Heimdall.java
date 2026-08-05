import data.custom.NumberImage;
import trainer.FeedForwardTrainer;

import static utils.NumberUtils.getAllImgs;
import static utils.NumberUtils.getRandomImgs;

public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		numberTrain();
	}
	
	public static void numberTrain() throws Exception {
		NumberImage[] allImages = getAllImgs("./src/main/resources/numbers/");
		
		NumberImage[] images = getRandomImgs(allImages, 1000, 123);
		float[][] targets = new float[images.length][],
				inputs = new float[images.length][];
		int[] outputs = new int[images.length];
		
		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i];
			inputs[i] = image.to1D();
			targets[i] = image.toTarget();
			outputs[i] = image.value();
		}
		
		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer().addLogger();

		//.loadBestAgent("./src/training-results/35");
		
		for (int generation = 1; generation <= 20000; generation++) {
			feedForwardTrainer.regularTrain(
					inputs,
					targets,
					outputs,
					.01f,
					generation
			);
		}
		
		System.out.println("Best Score: " + feedForwardTrainer.getBestScore());
		
//		trainer.saveAgent("agent");
	}
}
