package trainer;

import data.base.DataPoint;
import data.base.DataSet;
import model.NeuralNetwork;
import utils.DataUtils;

import java.text.DecimalFormat;

/**
 * A class representing a trainer object that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #addLogger()
 * @see #trainAgent(DataSet)
 */
public class FeedForwardTrainer extends AbstractTrainer{
	private final NeuralNetwork agent;
	private final float learningRate;
	private final boolean showErrorRate;


	public FeedForwardTrainer(int[] layerLengths, float learningRate, boolean showErrorRate, int seed){
		agent = new NeuralNetwork(layerLengths, seed);
		this.learningRate = learningRate;
		this.showErrorRate = showErrorRate;
	}


	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 */
	private int trainAgentRound(DataSet dataSet) {
		int score = 0;
		float errorSum = 0;
		
		for (DataPoint dataPoint : dataSet.dataPoints()) {
			float[] calcOutputs = agent.calcOutputs(dataPoint.getInputs());
			
			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++) {
				if (calcOutputs[j] > calcOutputs[maxIndex]) {
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			if (maxIndex == dataPoint.getTargetResult())
				score++;

			errorSum += DataUtils.getAverage(agent.backProp(dataPoint.getTargetValues()));
		}
		agent.applyWeightsChange(learningRate / dataSet.getSize());

		//System.out.println("MSE | " + Arrays.toString(MSE));
		if (showErrorRate){
			System.out.println("Error rate " + errorSum / dataSet.getSize());
		}
		
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
