import model.NeuralNetwork;
import utils.DataLogger;

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
 * @see #evolutionTrain(float[][], float[][], int[], float, int)
 * @see #saveAgent(String)
 * @see #loadAgent(String)
 */
public class Trainer {
	private final AtomicReference<NeuralNetwork> bestAgent;
	private final NeuralNetwork[] agents;
	private final AtomicLong bestScore;
	private DataLogger logger;
	
	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) {
		bestAgent = new AtomicReference<>(new NeuralNetwork(layerLengths, 123));
		bestScore = new AtomicLong(0);
		logger = null;
		agents = new NeuralNetwork[agentsPerRound];
		for (int i = 0; i < agentsPerRound; i++)
			agents[i] = new NeuralNetwork(layerLengths, 123);
	}
	
	/**
	 * initializes a {@link DataLogger} object to the current Trainer object.
	 *
	 * @return current object, allowing for inheritance chain and one-line setup
	 * @throws Exception if error occurs in {@link DataLogger#initLogger()}
	 */
	public Trainer addLogger() throws Exception {
		logger = new DataLogger("./src/training-results");
		logger.initLogger();
		return this;
	}
	
	/**
	 * Loads an agent from a serialized file.
	 * <p>
	 * Agent can be saved using {@link #saveAgent(String)}.
	 *
	 * @param path path to serialized file
	 * @return current trainer object, allowing for inheritance chain and one-line setup.
	 */
	public Trainer loadAgent(String path) {
		NeuralNetwork loaded = logger.loadAgent(path);
		bestAgent.set(loaded);
		Arrays.fill(agents, loaded);
		return this;
	}
	
	/**
	 * Saves an agent to a serialized object.
	 * <p>
	 * Agent can be loaded using {@link #loadAgent(String)}.
	 *
	 * @param agentName name of serialized agent file
	 */
	public void saveAgent(String agentName) {
		logger.saveAgent(bestAgent.get(), agentName);
	}
	
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
			
			float[] outputErrors = agent.backProp(targets[i]);
			for (int j = 0; j < outputErrors.length; j++)
				MSE[j] += (float) Math.pow(outputErrors[j], 2);
		}
		agent.applyWeights(learningRate);
		
		for (int i = 0; i < MSE.length; i++)
			MSE[i] /= inputs.length;

        System.out.println("MSE | " + Arrays.toString(MSE));
		
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
		int score = trainAgent(agents[0], inputs, targets, outputs, learningRate);
		
		float percent = (float) score / inputs.length * 100;
		String formatted = new DecimalFormat("###.##").format(percent);
		
		System.out.println("Generation: " + generationNum + " | Best: [" +
				                   score + "/" + inputs.length + "] (" + formatted + "%)");
	}
	
	public float getBestScore() {
		return bestScore.get();
	}
}
