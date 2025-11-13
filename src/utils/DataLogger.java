package utils;

import model.Layer;
import model.NeuralNetwork;
import model.Neuron;

import java.io.*;

public class DataLogger {
	private final File data, weights;
	
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
		float[][] allWeights = new float[layers.length][];
		
		for (int i = 0; i < layers.length; i++) {
			Layer layer = layers[i];
			for (int j = 0; j < layer.getNumNeurons(); j++) {
				Neuron neuron = network.getNeuron(i, j);
				allWeights[i] = neuron.getWeights();
			}
		}
		
		FileWriter writer = new FileWriter(weights);
		for (int i = 1; i < allWeights.length; i++) {
			float[] allWeight = allWeights[i];
			for (float v : allWeight)
				writer.write(v + ",");
			writer.write("\n");
		}
	}
}
