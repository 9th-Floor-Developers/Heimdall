import data.base.DataPoint;
import data.base.DataSet;
import data.custom.NumberImage;
import trainer.FeedForwardTrainer;

import java.util.ArrayList;
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
		
		NumberImage[] images = getRandomImgs(allImages, 2000, 123);

		ArrayList<DataPoint> dataPoints = new ArrayList<>();
		
		for (NumberImage image : images) {
			dataPoints.add(image.getDataPoint());
		}
		DataSet dataSet = new DataSet(dataPoints);

		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
			// number of agents per round, more possibilities to evolve
			new int[] {  // layers format
				dataSet.getInputLength(),  // input layer - must match input count
				100,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
				dataSet.getOutputLength()  // output layer - number of possible answers (0.0-1.0 inclusive)
			},
				0.001f,
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
