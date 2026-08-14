import core.data.CustomDataSetLoader;
import core.data.DataSet;
import core.trainers.FeedForwardTrainer;
import numberrecognizer.NumberImageLoader;

import java.util.List;


public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		numberTrain();
	}
	
	public static void numberTrain() throws Exception {
		/*
		To create a dataset, extend the AbstractDataSetLoader class
		AbstractDataSetLoader provides ability to adjust the dataset used in training and testing

		Here is an example using NumberImageLoader a loader from our number recognizer library:
		 */
		DataSet dataSet = NumberImageLoader.createLoader()
				.setSrc("./src/main/resources/numbers/")
				.setTrainingSize(10000)
				.setTestingSize(500)
				.load();
		/*
		Alternate example
		Loads 20K training data point and use remaining for testing:

		DataSet dataSet = NumberImageLoader.createLoader()
				.setSrc("./src/main/resources/numbers/")
				.setTrainingSize(20000)
				.setTestingSizeAsRemaining()
				.load();

		Quick way to create a new dataset:

		DataSet dataSet = CustomDataSetLoader.loadFromList(List.of(...))
				.setSrc("./src/main/resources/numbers/")
				.setTrainingSize(100)
				.setTestingSize(50)
				.load();
		 */


		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
			// number of agents per round, more possibilities to evolve
			new int[] {  // layers format
				30,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
				15,
			},
			5f,
			true
		).addLogger();//.loadBestAgent("./src/training-results/35");

		feedForwardTrainer.trainAgent(dataSet);
	}
}
