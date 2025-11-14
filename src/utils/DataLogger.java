package utils;

import model.NeuralNetwork;

import java.io.*;

public class DataLogger {
	private final String path;
	private File data;
	
	public DataLogger(String path) {
		File folder = new File(path);
		this.path = path + "/" + findNextSessionNumber(folder);
	}
	
	public void initLogger() throws IOException, InstantiationException {
		File folder = new File(path);
		if(!folder.mkdir())
			throw new InstantiationException("Failed To Create New Folder");
		
		data = new File(path + "/training-data.csv");
		
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
	
	public void logWeights(NeuralNetwork network) throws IOException, InstantiationException {
		File weights = new File(path + "/weights.csv");
		if (!weights.createNewFile())
			throw new InstantiationException("Failed To Create New Weights File");
		
		FileWriter writer = new FileWriter(weights);
		for (float[][] layerWeights : network.getWeights()) {
			for (float[] neuronWeights : layerWeights) {
				for (float weight : neuronWeights)
					writer.write(weight + ",");
				writer.write("\n");  // one node weights per line
			}
			writer.write("\n");  // whitespace between layers
		}
		
		writer.close();
	}
	
	public void logBiases(NeuralNetwork network) throws IOException, InstantiationException {
		File biases = new File(path + "/biases.csv");
		if (!biases.createNewFile())
			throw new InstantiationException("Failed To Create New Biases File");
		
		FileWriter writer = new FileWriter(biases);
		for (float[] layerBias : network.getBiases()) {
			for (float neuronBias : layerBias)
				writer.write(neuronBias + ",");
			writer.write("\n");  // one layer per line
		}
		
		writer.close();
	}
	
	public void saveBestAgent(NeuralNetwork agent) {
		try (FileOutputStream fileOut = new FileOutputStream(path + "/agent.ser");
		     ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
			out.writeObject(agent);
			System.out.println("Agent Saved Successfully...");
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public NeuralNetwork loadBestAgent(String folder) {
		try (FileInputStream fileIn = new FileInputStream(folder + "/agent.ser");
		     ObjectInputStream in = new ObjectInputStream(fileIn)) {
			NeuralNetwork agent = (NeuralNetwork) in.readObject();
			System.out.println("Agent Loaded Successfully");
			return agent;
		} catch (IOException | ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
}
