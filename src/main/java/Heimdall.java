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
		
		NumberImage[] images = getRandomImgs(allImages, 200, 123);

		DataSet dataSet = new DataSet(new ArrayList<>(Arrays.asList(images)));

		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
			// number of agents per round, more possibilities to evolve
			new int[] {  // layers format
				100,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
			},
			0.5f,
			true,
			69
		).addLogger();//.loadBestAgent("./src/training-results/35");

		feedForwardTrainer.trainAgent(dataSet);
	}
}
