import data.base.DataPoint;
import data.base.DataSet;
import data.custom.NumberImage;
import trainer.FeedForwardTrainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
		
		NumberImage[] images = getRandomImgs(allImages, 200, 123);

		DataSet dataSet = new DataSet(new ArrayList<>(Arrays.asList(images)));

		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
			// number of agents per round, more possibilities to evolve
			new int[] {  // layers format
				dataSet.getInputLength(),  // input layer - must match input count
				100,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
				dataSet.getOutputLength()  // output layer - number of possible answers (0.0-1.0 inclusive)
			},
			0.5f,
			true,
			69
		).addLogger();//.loadBestAgent("./src/training-results/35");

		feedForwardTrainer.trainAgent(dataSet);

		//.loadBestAgent("./src/training-results/35");
		/*
		System.out.println("Best Score: " + feedForwardTrainer.getBestScore());
		
		trainer.saveAgent("agent");
		 */
	}
}
