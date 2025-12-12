package legacy.matrixices;

import java.util.Random;

/**
 * A class that represents a neural network and all the neurons within.
 */
public class NeuralNetwork {
	private float[][] neurons;
	public float[][][] weights;
	private int[] layers;
	
	
	
	public NeuralNetwork(int[] layers) {
		neurons = new float[layers.length][];
		weights = new float[layers.length - 1][][];
		this.layers = layers;

		
		for (int i = 0; i < layers.length; i++){
			neurons[i] = new float[layers[i]];
			
			if (i == layers.length - 1){
				continue;
			}
			
			weights[i] = new float[layers[i]][layers[i + 1]];
		}
	}
	
	public float[] calculate(float[] inputs){
		//System.out.println("AA" + neurons.length );
		neurons[0] = inputs;
		for (int i = 0; i < neurons.length - 1; i++){
			for (int j = 0; j < neurons[i].length; j++){
				for (int k = 0; k < weights[i][j].length; k++) {
					neurons[i + 1][k] += weights[i][j][k] * neurons[i][j];
					//System.out.println(neurons[i + 1][k]);
 				}
			}
		}
		
		return neurons[neurons.length - 1];
	}
	
	public NeuralNetwork evolve_all(float scale){
		Random ramdom = new Random();
		NeuralNetwork network = new NeuralNetwork(layers);
		
		for (int i = 0; i < weights.length; i++){
			for (int j = 0; j < weights[i].length; j++){
				for (int k = 0; k < weights[i][j].length; k++){
					network.weights[i][j][k] = weights[i][j][k] + ramdom.nextFloat(-scale, scale);
				}
			}
		}
		return network;
	}
	
	public NeuralNetwork evolve(float scale, int amount){
		NeuralNetwork network = new NeuralNetwork(layers);
		Random ramdom = new Random();
		for (int t = 0; t < amount; t++){
			int i = ramdom.nextInt(weights.length);
			
			int j = ramdom.nextInt(weights[i].length);
			
			int k = ramdom.nextInt(weights[i][j].length);
			
			network.weights[i][j][k] = weights[i][j][k] + ramdom.nextFloat(-scale, scale);
		}
		return network;
	}
}
