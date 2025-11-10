import model.NeuralNetwork;

import java.text.DecimalFormat;

/**
 * A class representing a trainer that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 * <p>
 *
 * @see #train(float[][], float[][], int[], float)
 */
public class Trainer {
	private final NeuralNetwork agent;
	
	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) {
		agent = new NeuralNetwork(layerLengths);
	}
	
	/**
	 * Trains a single generation of agents.
	 *
	 * @param inputs       values neural network is trained on
	 * @param targets      calculated values of output layer
	 * @param outputs      desired values of the output layer
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @see NeuralNetwork
	 */
	public void train(float[][] inputs, float[][] targets, int[] outputs, float learningRate) {
		float sumLosses = 0;
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
			
			agent.backProp(targets[i], learningRate);
			sumLosses += agent.totalLoss();
		}
		
		float percent = (float) score / inputs.length * 100;
		String formatted = new DecimalFormat("###.##").format(percent);
		System.out.println("Score: [" + score + "/" + inputs.length + "] (" + formatted + "%)");
	}
}
