import model.NeuralNetwork;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A class representing a trainer that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 * <p>
 *
// * @see #train(float[][], float[][], int[], float)
 */
public class Trainer {
	private final AtomicReference<NeuralNetwork> bestAgent;
	private final NeuralNetwork[] variants;
	private final AtomicLong bestScore;
    private NeuralNetwork agent;
	
	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) {
		bestAgent = new AtomicReference<>(new NeuralNetwork(layerLengths));
		variants = new NeuralNetwork[agentsPerRound];
		bestScore = new AtomicLong(0);
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
	private void trainAgent(NeuralNetwork agent, float[][] inputs, float[][] targets,
	                         int[] outputs, float learningRate) {
		float[] MSE = new float[targets[0].length];
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
			
			float[] outputErrors = agent.backProp(targets[i], learningRate);
			for (int j = 0; j < outputErrors.length; j++){;
                MSE[j] += (float) Math.pow(outputErrors[j], 2);
            }
		}
        agent.applyWeights(inputs.length);

        for (int i = 0; i < MSE.length; i++){
            MSE[i] /= inputs.length;
        }

        float percent = (float) score / inputs.length * 100;
        String formatted = new DecimalFormat("###.##").format(percent);
        System.out.println("Score: [" + score + "/" + inputs.length + "] (" + formatted + "%)");
        System.out.println("Loss : [" + Arrays.toString(MSE) + "]");
	}
	
	public void train(float[][] inputs, float[][] targets, int[] outputs,
	                  float learningRate, int generationNum) throws InterruptedException {
		//ArrayList<Thread> threads = new ArrayList<>();

        System.out.println("Generation: " + generationNum);
        trainAgent(agent, inputs, targets, outputs, learningRate);

        /*
        for (int i = 0; i < variants.length; i++){
            float variantLoss = trainAgent(variants[i], inputs, targets, outputs, learningRate);
            if (variantLoss < loss){
                System.out.println("EVOLUTION BETTER");
            }
        }

         */

        /*
		for (NeuralNetwork variant : variants) {
			Thread thread = new Thread(() -> {
				float score = trainAgent(variant, inputs, targets, outputs, learningRate);
				if (score > bestScore.get()) {
					bestScore.set((long) score);
					bestAgent.set(variant);
                    System.out.println("Best replaced");
				}
			});
			thread.start();
			threads.add(thread);
		}
		
		for (Thread thread : threads)
			thread.join();
         */
	}
}
