import core.data.DataSet;
import numberrecognizer.NumberImage;
import core.trainers.FeedForwardTrainer;
import numberrecognizer.NumberImageLoader;



public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		numberTrain();
	}
	
	public static void numberTrain() throws Exception {
		DataSet dataSet = NumberImageLoader.createLoader()
				.setSrc("./src/main/resources/numbers/")
				.setTrainingSize(1000)
				.setTestingSize(100)
				.load();

		FeedForwardTrainer feedForwardTrainer = (FeedForwardTrainer) new FeedForwardTrainer(
			// number of agents per round, more possibilities to evolve
			new int[] {  // layers format
				30,  // hidden layer - number of middle layer nodes, more opportunities per agent to learn
			},
			5f,
			true
		).addLogger();//.loadBestAgent("./src/training-results/35");

		feedForwardTrainer.trainAgent(dataSet);
	}
}
