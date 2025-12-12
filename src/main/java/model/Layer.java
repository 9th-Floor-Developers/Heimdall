package model;

import java.io.Serializable;
import java.util.Random;

/**
 * A class that represents a single layer containing a set number of
 * {@link Neuron} objects.
 * <p>
 * A {@link NeuralNetwork} object contains at least 2 layers.
 */
public class Layer implements Serializable {
	private final Neuron[] neurons;
	
	/**
	 * Creates a layer of neurons and initializes all weights within except for input layer.
	 * <p>
	 * Input layer has no previous layer, so no weights are initialized.
	 *
	 * @param layerNum   layer index of current layer, used to initialize weights using previous layer
	 * @param network    {@link NeuralNetwork} object that layer is located in, used to find previous layer
	 * @param numNeurons number of {@link Neuron} objects to include in layer, output layer should
	 *                   be number of possible answers for all input values
	 * @param random An instance of random, to keep neural network constants
	 */
	public Layer(int layerNum, NeuralNetwork network, int numNeurons, Random random) {
		neurons = new Neuron[numNeurons];
		
		for (int i = 0; i < numNeurons; i++) {
			if (layerNum == 0) {
				Neuron neuron = new Neuron(0);
				neurons[i] = neuron;
				continue;
			}
			
			int numWeights = network.getLayer(layerNum - 1).getNumNeurons();
			neurons[i] = new Neuron(numWeights).initWeights(random);
		}
	}
	
	/**
	 * Calculates errors in current object based on {@link Neuron} objects within.
	 *
	 * @param error  difference between layer value and target value;
	 *               smaller error means better accuracy;
	 *               calculated using {@link Neuron#getError()}
	 * @param weights a singular weight value in next layer;
	 *               calculated using {@link Neuron#getWeight(int)}
	 */
	public void calcErrors(float error, float[] weights) {
        for (int i = 0; i < getNeurons().length; i++) {
            Neuron neuron = getNeurons()[i];
            neuron.addError(error * weights[i]);
        }
	}
	
	// region Getters/Setters
	public Neuron[] getNeurons() {
		return neurons;
	}
	
	public void setNeurons(Neuron[] neurons) {
		for (int i = 0; i < this.neurons.length; i++)
			setNeuron(i, neurons[i]);
	}
	
	
	public int getNumNeurons() {
		return neurons.length;
	}
	
	
	public Neuron getNeuron(int idx) {
		return neurons[idx];
	}
	
	public void setNeuron(int idx, Neuron neuron) {
		neurons[idx] = neuron;
	}
	
	
	public float[][] getWeights() {
		int numNeurons = getNumNeurons();
		float[][] weights = new float[numNeurons][];
		for (int i = 0; i < numNeurons; i++)
			weights[i] = getNeuron(i).getWeights();
		return weights;
	}
	
	public void setWeights(float[][] weights) {
		for (int i = 0; i < getNumNeurons(); i++)
			getNeuron(i).setWeights(weights[i]);
	}
	
	
	public float[] getBiases() {
		int numNeurons = getNumNeurons();
		float[] biases = new float[numNeurons];
		for (int i = 0; i < numNeurons; i++)
			biases[i] = getNeuron(i).getBias();
		return biases;
	}
	
	public void setBiases(float[] biases) {
		for (int i = 0; i < getNumNeurons(); i++)
			getNeuron(i).setBias(biases[i]);
	}
	
	
	public float[] getValues() {
		int numNeurons = getNumNeurons();
		float[] values = new float[numNeurons];
		for (int i = 0; i < numNeurons; i++)
			values[i] = getNeuron(i).getValue();
		return values;
	}
	
	public void setValues(float[] values) {
		for (int i = 0; i < getNumNeurons(); i++)
			getNeuron(i).setValue(values[i]);
	}
	
	
	public float[] getErrors() {
		int numNeurons = getNumNeurons();
		float[] errors = new float[numNeurons];
		for (int i = 0; i < numNeurons; i++)
			errors[i] = getNeuron(i).getError();
		return errors;
	}
	
	public void setErrors(float[] errors) {
		for (int i = 0; i < getNumNeurons(); i++)
			getNeuron(i).setError(errors[i]);
	}
	// endregion
}
