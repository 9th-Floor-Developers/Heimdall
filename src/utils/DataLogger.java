package utils;

import model.Layer;
import model.NeuralNetwork;

import java.io.*;
import java.util.Arrays;

public class DataLogger {
	private final File data, weights, biases;
	
	public DataLogger() throws IOException, InstantiationException {
		String folder_path = "./src/training-results";
		File trainingPath = new File(folder_path);
		
		if (!trainingPath.exists())
			throw new FileNotFoundException("Training results folder does not exist: " + trainingPath);
		
		File folder = new File(folder_path + "/" + findNextSessionNumber(trainingPath));
		if (!folder.mkdir())
			throw new InstantiationException("Failed To Create Directory.");
		
		data = new File(folder.getPath() + "/training-data.csv");
		if (!data.createNewFile())
			throw new InstantiationException("Failed To Create New Training Data File");
		weights = new File(folder.getPath() + "/weights.csv");
		if (!weights.createNewFile())
			throw new InstantiationException("Failed To Create New Weights File");
		biases = new File(folder.getPath() + "/biases.csv");
		if (!biases.createNewFile())
			throw new InstantiationException("Failed To Create New Biases File");
		
		FileWriter writer = new FileWriter(data);
		writer.write("Generation, Score, Total, Percent\n");
		writer.close();
	}
	
	private int findNextSessionNumber(File trainingPath) {
		int max = 0;
		for (String s : trainingPath.list()) {
			int number = Integer.parseInt(s);
			if (number > max)
				max = number;
		}
		
		return max + 1;
	}
	
	public void log(int generationNum, float score, int inputsLength, String formatted) throws IOException {
		FileWriter writer = new FileWriter(data, true);
		writer.write(generationNum + "," + score + "," + inputsLength + "," + formatted + "\n");
		writer.close();
	}
	
	public void logWeights(NeuralNetwork network) throws IOException {
		Layer[] layers = network.getLayers();
		float[][][] allWeights = new float[layers.length - 1][][];
		
		for (int j = 1; j < layers.length; j++) {  // ignore input layer
			int numNeurons = layers[j].getNumNeurons();
			allWeights[j - 1] = new float[numNeurons][];
			for (int i = 0; i < numNeurons; i++)
				allWeights[j - 1][i] = network.getNeuron(j, i).getWeights();
		}
		
		FileWriter writer = new FileWriter(weights, true);
		for (float[][] layerWeights : allWeights) {
			for (float[] neuronWeights : layerWeights) {
				for (float weight : neuronWeights)
					writer.write(weight + ",");
				writer.write("\n");  // one node weights per line
			}
			writer.write("\n");  // whitespace between layers
		}
		
		writer.close();
	}
	
	public void logBiases(NeuralNetwork network) throws IOException {
		Layer[] layers = network.getLayers();
		float[][] allBiases = new float[layers.length - 1][];
		
		for (int j = 1; j < layers.length; j++) {  // ignore input layer
			int numNeurons = layers[j].getNumNeurons();
			allBiases[j - 1] = new float[numNeurons];
			for (int i = 0; i < numNeurons; i++)
				allBiases[j - 1][i] = network.getNeuron(j, i).getBias();
		}
		
		FileWriter writer = new FileWriter(biases, true);
		for (float[] layerBias : allBiases) {
			for (float neuronBias : layerBias)
				writer.write(neuronBias + ",");
			writer.write("\n");  // one layer per line
		}
		
		writer.close();
	}
}
