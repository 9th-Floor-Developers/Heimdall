package core.trainers;

import core.data.DataPoint;
import core.data.DataSet;
import core.neuralnetwork.NeuralNetwork;
import core.utils.DataUtils;

import java.io.IOException;

/**
 * A class representing a trainer object that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #addLogger()
 * @see #trainAgent(DataSet)
 */
public class FeedForwardTrainer extends AbstractTrainer {
	private final int[] hiddenLayerLengths;
	private final float learningRate;
	private final boolean showErrorRate, focusOutput;
	private final int roundAmount;
	
	
	public FeedForwardTrainer(int[] hiddenLayerLengths, float learningRate, boolean showErrorRate, int roundAmount, boolean focusOutput) {
		this.hiddenLayerLengths = hiddenLayerLengths;
		this.learningRate = learningRate;
		this.showErrorRate = showErrorRate;
		this.roundAmount = roundAmount;
		this.focusOutput = focusOutput;
	}
	
	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 */
	private TrainingRoundResult trainAgentRound(DataSet dataSet, NeuralNetwork agent, int round) {
		int score = 0;
		float errorSum = showErrorRate ? 0 : -1;
		
		for (DataPoint dataPoint : dataSet.getTrainingDataPoints()) {
			float[] calcOutputs = agent.calcOutputs(dataPoint.inputs(), focusOutput);
			
			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++) {
				if (calcOutputs[j] > calcOutputs[maxIndex]) {
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			
			if (maxIndex == dataPoint.targetResult())
				score++;
			
			if (showErrorRate)
				errorSum += DataUtils.getAverage(agent.backProp(dataPoint.getTargetValues()));
		}
		
		agent.applyWeightsChange(learningRate / dataSet.getTrainingSize());
		
		float testScoreFactor = -1;
		if (round % testPerRoundAmount == 0)
			testScoreFactor = agent.testAgent(dataSet, focusOutput);
		
		return new TrainingRoundResult(
				round,
				(float) score / dataSet.getTrainingSize(),
				testScoreFactor,
				errorSum / dataSet.getTrainingSize()
		);
	}
	
	@Override
	public void trainAgent(DataSet dataSet) throws IOException {
		NeuralNetwork agent = new NeuralNetwork(dataSet.getLayerLengths(hiddenLayerLengths));
		
		//System.out.println("=========== Initial testing =============");
		agent.testAgent(dataSet, focusOutput);
		
		for (int round = 1; round <= roundAmount; round++) {
			TrainingRoundResult trainingRoundResult = trainAgentRound(dataSet, agent, round);
			trainingRoundResults.add(trainingRoundResult);
			
			if (logger != null)
				logger.logRound(round, trainingRoundResult.trainingScorePercent());
			
			printTrainingResults(round);
		}
	}
}
