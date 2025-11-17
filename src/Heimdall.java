import datasets.BasicDataSets;
import datasets.GeneratedDataSets;
import model.data.NumberImage;

import java.util.Random;

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
        NumberImage[] allImages = getAllImgs("./src/datasets/numbers/");

        NumberImage[] images = getRandomImgs(allImages, 100, 67);
        float[][] targets = new float[images.length][],
                inputs = new float[images.length][];
        int[] outputs = new int[images.length];

        for (int i = 0; i < images.length; i++) {
            NumberImage image = images[i];
            inputs[i] = image.to1D();
            targets[i] = image.toTarget();
            outputs[i] = image.value();
        }

        Trainer trainer = new Trainer(
                10,  // number of agents per round, more possibilities to evolve
                new int[] {  // layers format
                        inputs[0].length,  // input layer - must match input count
                        100,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
                        targets[0].length  // output layer - number of possible answers (0.0-1.0 inclusive)
                }
        ).addLogger();//.loadBestAgent("./src/training-results/35");

        for (int generation = 1; generation <= 100; generation++) {
            trainer.regularTrain(
                    inputs,
                    targets,
                    outputs,
                    .01f,
                    generation
            );
        }

        System.out.println("Best Score: " + trainer.getBestScore());

        //trainer.saveBestAgent();
    }
}
