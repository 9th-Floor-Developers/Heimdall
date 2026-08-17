package core.neuralnetwork;

import core.data.DataPoint;
import core.data.DataSet;
import core.utils.DataLogger;
import core.utils.DataUtils;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Comparator;
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
public class NeuralNetwork implements Serializable {
	private final Layer[] layers;
	private final int[] layerLengths;

	/**
	 * Creates a neural network and initializes all layers, neurons, and weights within
	 *
	 * @param layerLengths array containing number of {@link Neuron} objects in each {@link Layer},
	 *                     {@code layerLengths.length} should be total number of layers in network.k
	 */
	public NeuralNetwork(int[] layerLengths) {
        Random random = new Random(DataUtils.universalSeed);

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
	public float[] calcOutputs(float[] inputs, boolean focusOutputs) {
		int outputLayerIdx = layers.length - 1;
		float[] outputs = new float[layers[outputLayerIdx].getNumNeurons()];
		
		for (int i = 0; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				Neuron neuron = neurons[j];
				if (i == 0) {
					neuron.setValue(inputs[j]);
					continue;
				}

				neuron.calcValue(layers[i - 1]);
			}

			if (i == outputLayerIdx) {
				if (focusOutputs){
					focusOutputs(neurons);
				}

				for (int j = 0; j < neurons.length; j++) {
					outputs[j] = neurons[j].getValue();
				}
			}
		}
		
		return outputs;
	}

	public void focusOutputs(Neuron[] outputNeurons){
		Neuron maxNeron = Arrays.stream(outputNeurons).max(Comparator.comparingDouble(Neuron::getValue)).orElseThrow();
		for (Neuron outputNeron: outputNeurons){
            if (outputNeron == maxNeron) {
                outputNeron.setValue(1);
            }
			else {
                outputNeron.setValue(0);
            }
        }
	}

	/**
	 * Apply back propagation process to neural network.
	 * This requires the value so {@link NeuralNetwork#calcOutputs(float[], boolean)} needs to be run first
	 *
	 * @param target desired output values
	 * @see Layer
	 * @see Neuron
	 */
	public float[] backProp(float[] target) {
		float[] outputErrors = new float[target.length];
		
		for (int i = layers.length - 1; i >= 1; i--) {
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == layers.length - 1) {  // output layer
					neuron.calcError(target[j]);
					outputErrors[j] = Math.abs(neuron.getError());
				}
				
				Layer prevLayer = layers[i - 1];
				prevLayer.calcErrors(neuron.getError(), neuron.getWeights());

				neuron.calcWeightChange(prevLayer);
			}
		}
		
		return outputErrors;
	}
	
	/**
	 * Runs {@link Neuron#applyWeightChange(float)} function to all {@link Neuron} objects in network.
	 * Is used after the changes have been calculated from running {@link NeuralNetwork#backProp(float[])}
	 *
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @see Layer
	 */
	public void applyWeightsChange(float learningRate) {
		for (int i = 0; i < layers.length; i++)
			for (int j = 0; j < layers[i].getNumNeurons(); j++)
				getNeuron(i, j).applyWeightChange(learningRate);
	}

	public float testAgent(DataSet dataSet, boolean focusOutputs){
		int score = 0;

		for (DataPoint dataPoint : dataSet.testingDataPoints()) {
			float[] calcOutputs = calcOutputs(dataPoint.getInputs(), focusOutputs);

			int maxIndex = 0;
			for (int j = 0; j < calcOutputs.length; j++) {
				if (calcOutputs[j] > calcOutputs[maxIndex]) {
					calcOutputs[maxIndex] = calcOutputs[j];
					maxIndex = j;
				}
			}
			if (maxIndex == dataPoint.getTargetResult())
				score++;
		}

		float percent = (float) score / dataSet.getTestingSize() * 100;
		String formatted = new DecimalFormat("###.##").format(percent);

		System.out.println("Testing: [" + score + "/" + dataSet.getTestingSize() + "] (" + formatted + "%)");

		return percent;
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
