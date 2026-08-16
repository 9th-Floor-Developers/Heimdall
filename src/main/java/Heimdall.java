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
				.setRandomTrainingSize(5)
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
				// number of agents per round, more possibilities to evolve
				new int[] {  // layers format
						30,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
						15,
				},
				5f,
				true,
				600,
				false
		).addLogger();

		feedForwardTrainer.trainAgent(dataSet);
	}
}
