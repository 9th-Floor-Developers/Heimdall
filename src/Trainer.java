import model.NeuralNetwork;
import model.Neuron;
import utils.DataLogger;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

// TODO: update docstrings

/**
 * A class representing a trainer that contains an array of agents
 * ({@link NeuralNetwork} objects) and all methods required to train them.
 * <p>
 *
// * @see #train(float[][], float[][], int[], float)
 */
public class Trainer {
	private final AtomicReference<NeuralNetwork> bestAgent;
	private final NeuralNetwork[] agents;
	private final AtomicLong bestScore;
	private final DataLogger logger;
	
	/**
	 * Initialize trainer and all agents ({@link NeuralNetwork} objects) within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing lengths of each layer
	 *                       ({@code layerLengths.length} will be used to create number of layers)
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) throws Exception {
		bestAgent = new AtomicReference<>(new NeuralNetwork(layerLengths));
		bestScore = new AtomicLong(0);
		logger = new DataLogger();
		agents = new NeuralNetwork[agentsPerRound];
		for (int i = 0; i < agentsPerRound; i++)
			agents[i] = new NeuralNetwork(layerLengths);
	}
	
	public void loadData(String weightsPath, String biasesPath) throws FileNotFoundException {
		File weights = new File(weightsPath),
			biases = new File(biasesPath);
		
		String delimiter = ",";
		
		ArrayList<ArrayList<ArrayList<Float>>> allWeights = readWeights(weights, delimiter);
		ArrayList<ArrayList<Float>> allBiases = readBiases(biases, delimiter);
		
		int[] layerLengths = agents[0].getLayerLengths();
		NeuralNetwork savedAgent = new NeuralNetwork(layerLengths);
		for (int i = 1; i < layerLengths.length; i++) {
			for (int j = 0; j < layerLengths[i]; j++) {
				Neuron neuron = savedAgent.getNeuron(i, j);
				for (int k = 0; k < allWeights.get(i - 1).get(j).size(); k++)
					neuron.setWeight(k, allWeights.get(i - 1).get(j).get(k));
				neuron.setBias(allBiases.get(i - 1).get(j));
			}
		}
		
		Arrays.fill(agents, savedAgent);
	}
	
	private static ArrayList<ArrayList<ArrayList<Float>>> readWeights(File file, String delimiter) throws FileNotFoundException {
		ArrayList<ArrayList<ArrayList<Float>>> allData = new ArrayList<>();
		ArrayList<ArrayList<Float>> layerData = new ArrayList<>();
		Scanner reader = new Scanner(file);
		while (reader.hasNextLine()) {
			String line = reader.nextLine();
			if (line.isEmpty()) {
				allData.add(layerData);
				layerData = new ArrayList<>();
				continue;
			}
			String[] data = line.split(delimiter);
			
			ArrayList<Float> nodeData = new ArrayList<>();
			for (String datum : data)
				nodeData.add(Float.parseFloat(datum));
			
			layerData.add(nodeData);
		}
		
		return allData;
	}
	
	private static ArrayList<ArrayList<Float>> readBiases(File file, String delimiter) throws FileNotFoundException {
		ArrayList<ArrayList<Float>> allData = new ArrayList<>();
		Scanner reader = new Scanner(file);
		while (reader.hasNextLine()) {
			ArrayList<Float> layerData = new ArrayList<>();
			String line = reader.nextLine();
			String[] data = line.split(delimiter);
			
			for (String datum : data)
				layerData.add(Float.parseFloat(datum));
			
			allData.add(layerData);
		}
		
		return allData;
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
	private float trainAgent(NeuralNetwork agent, float[][] inputs, float[][] targets,
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
        agent.applyWeights(learningRate);

        for (int i = 0; i < MSE.length; i++){
            MSE[i] /= inputs.length;
        }

		return score;
	}
	
	public void train(float[][] inputs, float[][] targets, int[] outputs,
	                  float learningRate, int generationNum) throws Exception {
		// TODO: add multithreading
		float[] scores = new float[agents.length];
		for (int i = 0; i < agents.length; i++)
			scores[i] = trainAgent(agents[i], inputs, targets, outputs, learningRate);
		
		int bestIndex = 0;
		for (int i = 0; i < scores.length; i++)
			if (scores[i] > scores[bestIndex])
				bestIndex = i;

		float bestRoundScore = scores[bestIndex];
		float percent = bestRoundScore / inputs.length * 100;
		String formatted = new DecimalFormat("###.##").format(percent);
		
		System.out.println("Generation: " + generationNum + " | Best: [" + bestRoundScore + "/" + inputs.length + "] (" + formatted + "%)");
		logger.log(generationNum, bestRoundScore, inputs.length, formatted);
		
		if (bestRoundScore > bestScore.get()) {
			bestScore.set((long) bestRoundScore);
			bestAgent.set(agents[bestIndex]);
		}
		
		NeuralNetwork bestRoundAgent = agents[bestIndex];
		for (int i = 0; i < agents.length; i++)
			agents[i] = bestRoundAgent.evolve(learningRate);
	}
	
	public float getBestScore() {
		return bestScore.get();
	}
	
	/**
	 * Logs all weights of the best agent in weights.csv
	 * <p>
	 * Should only be run after the final training session.
	 *
	 * @throws IOException if file logging fails
	 * @see DataLogger#logWeights(NeuralNetwork)
	 */
	public void logWeights() throws IOException {
		logger.logWeights(bestAgent.get());
	}
	
	/**
	 * Logs all biases of the best agent in biases.csv
	 * <p>
	 * Should only be run after the final training session.
	 *
	 * @throws IOException if file logging fails
	 * @see DataLogger#logBiases(NeuralNetwork)
	 */
	public void logBiases() throws IOException {
		logger.logBiases(bestAgent.get());
	}
}
