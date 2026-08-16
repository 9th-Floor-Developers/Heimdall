package core.trainers;

import core.data.DataPoint;
import core.data.DataSet;
import core.neuralnetwork.NeuralNetwork;
import core.utils.DataUtils;

import java.io.IOException;
import java.text.DecimalFormat;

/**
 * A class representing a trainer object that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #addLogger()
 * @see #trainAgent(DataSet)
 */
public class FeedForwardTrainer extends AbstractTrainer{
	private final int[] hiddenLayerLengths;
	private final float learningRate;
	private final boolean showErrorRate;
	private final int roundAmount;
	private final boolean focusOutput;


	public FeedForwardTrainer(int[] hiddenLayerLengths, float learningRate, boolean showErrorRate, int roundAmount, boolean focusOutput){
		this.hiddenLayerLengths = hiddenLayerLengths;
		this.learningRate = learningRate;
		this.showErrorRate = showErrorRate;
		this.roundAmount = roundAmount;
		this.focusOutput = focusOutput;
	}

	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 */
	private int trainAgentRound(NeuralNetwork agent, DataSet dataSet) {
		int score = 0;
		float errorSum = 0;
		
		for (DataPoint dataPoint : dataSet.allTrainingDataPoints()) {
			float[] calcOutputs = agent.calcOutputs(dataPoint.getInputs(), focusOutput);
			
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
		agent.applyWeightsChange(learningRate / dataSet.getAllTrainingSize());

		//System.out.println("MSE | " + Arrays.toString(MSE));
		if (showErrorRate){
			System.out.println("Error rate " + errorSum / dataSet.getAllTrainingSize());
		}
		
		return score;
	}

	@Override
	public void trainAgent(DataSet dataSet) throws IOException {
		NeuralNetwork agent = new NeuralNetwork(dataSet.getLayerLengths(hiddenLayerLengths));

		System.out.println("=========== Initial testing =============");
		agent.testAgent(dataSet, focusOutput);

		for (int round = 1; round <= roundAmount; round++) {
			System.out.println("=========== Round: " + round + " =============");

			int score = trainAgentRound(agent, dataSet);

			float percent = (float) score / dataSet.getAllTrainingSize() * 100;
			String formatted = new DecimalFormat("###.##").format(percent);

			System.out.println("Training: [" + score + "/" + dataSet.getAllTrainingSize() + "] (" + formatted + "%)");

			float testPercent = agent.testAgent(dataSet, focusOutput);

			System.out.println("===================================");

			if (logger != null){
				logger.logRound(round, testPercent);
			}
		}
	}
}
