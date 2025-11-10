package model;

import java.util.Random;

/**
 * A class that represents a single neuron.
 *
 * @see Layer
 * @see NeuralNetwork
 */
public class Neuron {
	private final float[] weights;  // array of all weights for next layer
	private float value, bias, error;  // value = weights * value + bias
	
	/**
	 * Creates a neuron with an empty (uninitialized) weights array
	 * <p>
	 * Initialize weights using {@link #initWeights()}, {@link #setWeight(int, float)},
	 * or {@link #addWeight(int, float)}.
	 *
	 * @param weightsSize desired size of weights array, the number of nodes in previous layer (input layer has 0)
	 */
	public Neuron(int weightsSize) {
		weights = new float[weightsSize];
	}
	
	/**
	 * Initialize all weights with random float between -1 and 1
	 */
	public void initWeights() {
		Random random = new Random();
		bias = random.nextFloat(-1, 1);
		
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat(-1, 1);
	}
	
	/**
	 * Calculates value using equation: v = w * v + b
	 *
	 * @param prevLayer previous {@link Layer} of neurons, used to calculate current neuron's value
	 */
	public void calcValue(Layer prevLayer) {
		value = bias;
		error = 0;
		
		Neuron[] prevNeurons = prevLayer.getNeurons();
		for (int i = 0; i < prevNeurons.length; i++) {
			Neuron prevNeuron = prevNeurons[i];
			value += prevNeuron.value * getWeight(i);
		}
		
		value = sigmoid(value);
	}
	
	/**
	 * Calculates errors in previous error
	 *
	 * @param prevLayer layer to calculate errors for
	 */
	public void calcErrors(Layer prevLayer) {
		Neuron[] prevNeurons = prevLayer.getNeurons();
		for (int i = 0; i < prevNeurons.length; i++) {
			Neuron prevNeuron = prevNeurons[i];
			prevNeuron.error += error * getWeight(i);
		}
	}
	
	/**
	 * Calculates the gradient of the cost function to find the global minimum.
	 *
	 * @param learningRate rate to apply to weights (0.00-0.10)
	 * @param prevLayer    layer to get neuron weights from
	 */
	public void modifyWeights(float learningRate, Layer prevLayer) {
		Neuron[] prevNeurons = prevLayer.getNeurons();
		for (int i = 0; i < prevNeurons.length; i++) {
			Neuron prevNeuron = prevNeurons[i];
			float gradient = error * sigmoidDerivative(value);
			addWeight(i, learningRate * gradient * prevNeuron.getValue());
		}
		addBias(learningRate * error);
	}
	
	/**
	 * Calculates sigmoid function
	 *
	 * @param x value of x in sigmoid function
	 * @return output value of sigmoid function
	 */
	private float sigmoid(float x) {
		return (float) (1 / (1 + Math.exp(-x)));
	}
	
	/**
	 * Calculates derivative of sigmoid function
	 *
	 * @param y value of y in derivative of sigmoid function
	 * @return output value of derivative of sigmoid function
	 */
	private float sigmoidDerivative(float y) {
		return y * (1 - y);
	}
	
	public float getWeight(int idx) {
		return weights[idx];
	}
	
	public float[] getWeights() {
		return weights;
	}
	
	public int getNumWeights() {
		return weights.length;
	}
	
	public void setWeight(int idx, float weight) {
		weights[idx] = weight;
	}
	
	public void addWeight(int index, float value) {
		weights[index] += value;
	}
	
	public float getValue() {
		return value;
	}
	
	public void setValue(float value) {
		this.value = value;
	}
	
	public void resetValue() {
		value = 0;
	}
	
	public void addValue(float value) {
		this.value += value;
	}
	
	public float getBias() {
		return bias;
	}
	
	public void addBias(float bias) {
		this.bias += bias;
	}
	
	public float getError() {
		return error;
	}
	
	public void addError(float error) {
		this.error += error;
	}
	
	public void setError(float target) {
		this.error = target - value;
	}
}
