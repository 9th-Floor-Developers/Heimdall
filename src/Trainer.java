import model.Layer;
import model.NeuralNetwork;
import model.Neuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * A class representing a trainer that contains an array of agents
 * ({@link Neuron} objects) and all methods required to train them.
 *
// * @see #train(float[][], int[], int, float)
 */
public class Trainer {
	private NeuralNetwork agent;  // all agents to be used in training session

	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 * @throws Exception if {@code agentsPerRound} is not divisible by 4, meaning agents cannot be evenly distributed
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) throws Exception {
		agent = new NeuralNetwork(layerLengths);
	}
		
	public float train(float[][] inputs, float[][] targets, int[] outputs, float learningRate)  {
		System.out.println("Training...");
		
		float sumLosses = 0;
		int score = 0;
		
		for (int i = 0; i < inputs.length; i++) {
			float[] calcOutputs = agent.calculate(inputs[i]);
			
			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++){
				if (calcOutputs[j] > calcOutputs[maxIndex]){
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			if (maxIndex == outputs[i]){
				score++;
			}
			
			agent.backProp(targets[i], learningRate);
			sumLosses += agent.totalLoss();
		}
		
		System.out.println("Score [" + score + "] [" + score / inputs.length * 100 + "%]");
		return sumLosses / inputs.length;
	}
}
