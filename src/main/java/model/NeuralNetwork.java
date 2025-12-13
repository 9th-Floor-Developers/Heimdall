package model;

import utils.DataLogger;

import java.io.Serializable;
import java.util.Random;

/**
 * A class that represents a neural network and all the layers within.
 * Each {@link Layer} contains an array of {@link Neuron} objects.
 * <p>
 * Class can be serialized to save/load the best agent.
 *
 * @see DataLogger#saveAgent(NeuralNetwork, String)
 * @see DataLogger#loadAgent(String)
 */
public class NeuralNetwork implements Serializable, Cloneable {
	private final Layer[] layers;
	private final int[] layerLengths;
	
	// TODO: create alternative without seed parameter
	/**
	 * Creates a neural network and initializes all layers, neurons, and weights within
	 *
	 * @param layerLengths array containing number of {@link Neuron} objects in each {@link Layer},
	 *                     {@code layerLengths.length} should be total number of layers in network.
	 * @param seed         initial value of the internal state of the pseudorandom number generator used in
	 *                     {@link Neuron} objects inside this network
	 */
	public NeuralNetwork(int[] layerLengths, int seed) {
        Random random = new Random(seed);

		layers = new Layer[layerLengths.length];
		this.layerLengths = layerLengths;
		
		for (int i = 0; i < layerLengths.length; i++)
			layers[i] = new Layer(i, this, layerLengths[i], random);
	}
	
	/**
	 * Returns network determined values of output layer.
	 * <p>
	 * When return value is compared with definitive answer array, the accuracy of the network can be determined.
	 *
	 * @param inputs values neural network is trained on
	 * @return values of output layer, should be used to compare definitive answer array.
	 * @see Layer
	 * @see Neuron
	 */
	public float[] calcOutputs(float[] inputs) {
		int outputLayerIdx = layers.length - 1;
		float[] outputs = new float[layers[outputLayerIdx].getNumNeurons()];
		
		for (int i = 0; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == 0) {
					neuron.setValue(inputs[j]);
					continue;
				}
				
				neuron.calcValue(layers[i - 1]);
				if (i == outputLayerIdx)
					outputs[j] = neuron.getValue();
			}
		}
		
		return outputs;
	}
	
	/**
	 * Apply back propagation process to neural network.
	 *
	 * @param target desired output values
	 * @see Layer
	 * @see Neuron
	 */
	public float[] backProp(float[] target) {
		float[] outputError = new float[target.length];
		
		for (int i = layers.length - 1; i >= 1; i--) {
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == layers.length - 1) {  // output layer
					neuron.calcError(target[j]);
					outputError[j] = neuron.getError();
				}
				
				Layer prevLayer = layers[i - 1];
				prevLayer.calcErrors(neuron.getError(), neuron.getWeights());

				neuron.calcWeightChange(prevLayer);
			}
		}
		
		return outputError;
	}
	
	/**
	 * Runs {@link Neuron#applyWeightChange(float)} function to all {@link Neuron} objects in network.
	 *
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @see Layer
	 */
	public void applyWeightsChange(float learningRate) {
		for (int i = 0; i < layers.length; i++)
			for (int j = 0; j < layers[i].getNumNeurons(); j++)
				getNeuron(i, j).applyWeightChange(learningRate);
	}
	
	// region Getters/Setters
	public Layer[] getLayers() {
		return layers;
	}
	
	public int[] getLayerLengths() {
		return layerLengths;
	}
	
	public Layer getLayer(int idx) {
		return layers[idx];
	}
	
	public void setLayer(int idx, Layer layer) {
		layers[idx] = layer;
	}
	
	
	public Neuron getNeuron(int layer, int number) {
		return layers[layer].getNeuron(number);
	}
	
	public void setNeuron(int layer, int idx, Neuron neuron) {
		layers[layer].setNeuron(idx, neuron);
	}
	
	
	public Neuron[][] getNeurons() {
		Neuron[][] neurons = new Neuron[layers.length][];
		for (int i = 0; i < layers.length; i++)
			neurons[i] = layers[i].getNeurons();
		return neurons;
	}
	
	public void setNeurons(Neuron[][] neurons) {
		for (int i = 0; i < layers.length; i++)
			layers[i].setNeurons(neurons[i]);
	}
	
	
	public void setWeights(float[][][] weights) {
		for (int i = 1; i < layers.length; i++)
			layers[i].setWeights(weights[i - 1]);
	}
	
	public float[][][] getWeights() {
		float[][][] weights = new float[layers.length - 1][][];
		for (int i = 1; i < layers.length; i++)  // ignore input layer
			weights[i - 1] = layers[i].getWeights();
		return weights;
	}
	
	
	public void setBiases(float[][] biases) {
		for (int i = 1; i < layerLengths.length; i++)
			layers[i].setBiases(biases[i - 1]);
	}
	
	public float[][] getBiases() {
		float[][] biases = new float[layers.length - 1][];
		for (int i = 1; i < layers.length; i++)  // ignore input layer
			biases[i - 1] = layers[i].getBiases();
		return biases;
	}
	
	
	public float[][] getValues() {
		float[][] values = new float[layers.length - 1][];
		for (int i = 1; i < layers.length; i++)  // ignore input layer
			values[i - 1] = layers[i].getValues();
		return values;
	}
	
	public void setValues(float[][] values) {
		for (int i = 1; i < layerLengths.length; i++)
			layers[i].setValues(values[i - 1]);
	}
	
	
	public float[][] getErrors() {
		float[][] errors = new float[layers.length - 1][];
		for (int i = 1; i < layers.length; i++)  // ignore input layer
			errors[i - 1] = layers[i].getErrors();
		return errors;
	}
	
	public void setErrors(float[][] errors) {
		for (int i = 1; i < layerLengths.length; i++)
			layers[i].setErrors(errors[i - 1]);
	}
	// endregion
}
