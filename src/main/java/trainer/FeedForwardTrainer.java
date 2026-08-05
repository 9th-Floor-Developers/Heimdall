package trainer;

import model.NeuralNetwork;
import utils.DataLogger;
import utils.DataUtils;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A class representing a trainer object that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 *
 * @see #addLogger()
 * @see #getBestScore()
 * @see #regularTrain(float[][], float[][], int[], float, int)
 */
public class FeedForwardTrainer extends AbstractTrainer{
	/**
	 * Trains a single {@link NeuralNetwork} agent using the gradient decent algorithm with back propagation.
	 *
	 * @param agent        {@link NeuralNetwork} object to train with
	 * @param inputs       values neural network is trained on
	 * @param targets      calculated values of output layer
	 * @param outputs      desired values of the output layer
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @return number of data points the agent got correct
	 */
	private int trainAgent(NeuralNetwork agent, float[][] inputs, float[][] targets,
	                       int[] outputs, float learningRate, boolean showErrorRate) {
		int score = 0;
		
		for (int i = 0; i < inputs.length; i++) {
			float[] calcOutputs = agent.calcOutputs(inputs[i]);
			
			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++) {
				if (calcOutputs[j] > calcOutputs[maxIndex]) {
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			if (maxIndex == outputs[i])
				score++;
			
			float[] outputErrors = agent.backProp(targets[i]);
			if (showErrorRate){
				System.out.println("Error rate " + DataUtils.getAverage(outputErrors));
			}
		}
		agent.applyWeightsChange(learningRate);

		//System.out.println("MSE | " + Arrays.toString(MSE));
		
		return score;
	}
	
	/**
	 * Train a single agent using back propagation, single threaded.
	 * <p>
	 * Outputs generation information.
	 *
	 * @param inputs        values neural network is trained on
	 * @param targets       calculated values of output layer
	 * @param outputs       desired values of the output layer
	 * @param learningRate  difference to modify weights (0.0-0.5)
	 * @param generationNum current generation number, used only for displaying information
	 */
	public void regularTrain(float[][] inputs, float[][] targets, int[] outputs,
	                         float learningRate, int generationNum) {
		/*
		int score = trainAgent(agents[0], inputs, targets, outputs, learningRate, false);
		
		float percent = (float) score / inputs.length * 100;
		String formatted = new DecimalFormat("###.##").format(percent);
		
		System.out.println("Generation: " + generationNum + " | Best: [" +
				                   score + "/" + inputs.length + "] (" + formatted + "%)");

		 */
	}


	@Override
	public void trainAgent() {

	}

	@Override
	public float getBestScore() {
		return 0;
	}
}
