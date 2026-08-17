import core.data.DataSet;
import core.trainers.FeedForwardTrainer;
import numberrecognizer.NumberImageLoader;



public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		trainNumberRecognizer();
	}

	public static void trainNumberRecognizer() throws Exception {
		/*
		To create a dataset, extend the AbstractDataSetLoader class
		AbstractDataSetLoader provides ability to adjust the dataset used in training and testing

		Here is an example using NumberImageLoader a loader from our number recognizer library:
		 */
		DataSet dataSet = NumberImageLoader.createLoader()
				.setSrc("./src/main/resources/numbers/")
				.setAllTrainingSize(10000)
				.setRandomTrainingSize(100) //This is optional, you are able to make the model train on random sets of data points,
				//This means it loads 10k datapoints for training and pick a random 100 datapoints from that to train every round
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
				.setTrainingSize(100)
				.setTestingSize(50)
				.load();
		 */
		dataSet.printSize(); //Method just to double-check that the size is correct

		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
				new int[] {
					30,  //Hidden layer lengths - how deep can the agent think, for higher
					15,
				},
				5f,
				true,
				100,
				false
		)
		.addLogger() //Logs the result of every training round on to a file
		.setPrintPerRoundAmount(1); //Will print out the training results for every X round

		feedForwardTrainer.trainAgent(dataSet);
	}
}
