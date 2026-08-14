import data.base.DataSet;
import data.numberRecognizer.NumberImage;
import trainer.FeedForwardTrainer;

import java.util.ArrayList;
import java.util.Arrays;

import static data.numberRecognizer.NumberImageLoader.getAllImgs;
import static data.numberRecognizer.NumberImageLoader.getRandomImgs;

public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		numberTrain();
	}
	
	public static void numberTrain() throws Exception {
		NumberImage[] allImages = getAllImgs("./src/main/resources/numbers/");
		
		NumberImage[] trainingImages = getRandomImgs(allImages, 5000);
		NumberImage[] testingImages = getRandomImgs(allImages, 400);

		DataSet dataSet = new DataSet(new ArrayList<>(Arrays.asList(trainingImages)), new ArrayList<>(Arrays.asList(testingImages)));

		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
			// number of agents per round, more possibilities to evolve
			new int[] {  // layers format
				100,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
			},
			5f,
			true
		).addLogger();//.loadBestAgent("./src/training-results/35");

		feedForwardTrainer.trainAgent(dataSet);
	}
}
