package trainer;

import data.base.DataPoint;
import data.base.DataSet;
import model.NeuralNetwork;
import utils.DataUtils;

import java.text.DecimalFormat;
import java.util.Arrays;

/**
 * A class representing a trainer object that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #addLogger()
 * @see #trainAgent(DataSet)
 */
public class FeedForwardTrainer extends AbstractTrainer{
	public NeuralNetwork agent;
	public float learningRate;
	public boolean showErrorRate;


	public FeedForwardTrainer(int[] layerLengths, int seed){
		agent = new NeuralNetwork(layerLengths, seed);
	}


	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 */
	private int trainAgentRound(DataSet dataSet) {
		int score = 0;
		
		for (DataPoint dataPoint : dataSet.dataPoints()) {
			float[] calcOutputs = agent.calcOutputs(dataPoint.inputs());
			
			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++) {
				if (calcOutputs[j] > calcOutputs[maxIndex]) {
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			if (maxIndex == dataPoint.targetResult())
				score++;

			float[] outputErrors = agent.backProp(dataPoint.targetValues());
			if (showErrorRate){
				System.out.println("Error rate " + DataUtils.getAverage(outputErrors));
			}
		}
		agent.applyWeightsChange(learningRate);

		//System.out.println("MSE | " + Arrays.toString(MSE));
		
		return score;
	}

	@Override
	public void trainAgent(DataSet dataSet) {
		for (int generation = 1; generation <= 20000; generation++) {
			int score = trainAgentRound(dataSet);

			float percent = (float) score / dataSet.getSize() * 100;
			String formatted = new DecimalFormat("###.##").format(percent);

			System.out.println("Generation: " + generation + " | Best: [" +
					score + "/" + dataSet.getSize() + "] (" + formatted + "%)");
		}
	}
}
