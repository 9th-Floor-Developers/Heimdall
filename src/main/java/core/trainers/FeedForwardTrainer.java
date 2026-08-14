package core.trainers;

import core.data.DataPoint;
import core.data.DataSet;
import core.neuralnetwork.NeuralNetwork;
import core.utils.DataUtils;

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


	public FeedForwardTrainer(int[] hiddenLayerLengths, float learningRate, boolean showErrorRate){
		this.hiddenLayerLengths = hiddenLayerLengths;
		this.learningRate = learningRate;
		this.showErrorRate = showErrorRate;
	}


	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 */
	private int trainAgentRound(NeuralNetwork agent, DataSet dataSet) {
		int score = 0;
		float errorSum = 0;
		
		for (DataPoint dataPoint : dataSet.trainingDataPoints()) {
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
		agent.applyWeightsChange(learningRate / dataSet.getTrainingSize());

		//System.out.println("MSE | " + Arrays.toString(MSE));
		if (showErrorRate){
			System.out.println("Error rate " + errorSum / dataSet.getTrainingSize());
		}
		
		return score;
	}

	@Override
	public void trainAgent(DataSet dataSet) {
		NeuralNetwork agent = new NeuralNetwork(dataSet.getLayerLengths(hiddenLayerLengths));

		System.out.println("=========== Initial testing =============");
		agent.testAgent(dataSet);

		for (int round = 1; round <= 600; round++) {
			System.out.println("=========== Round: " + round + " =============");

			int score = trainAgentRound(agent, dataSet);

			float percent = (float) score / dataSet.getTrainingSize() * 100;
			String formatted = new DecimalFormat("###.##").format(percent);

			System.out.println("Training: [" + score + "/" + dataSet.getTrainingSize() + "] (" + formatted + "%)");

			agent.testAgent(dataSet);

			System.out.println("===================================");
		}
	}
}
