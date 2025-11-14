package model;

import java.io.Serializable;
import java.util.Random;

/**
 * A class that represents a single neuron located in a {@link Layer}
 * object, located in a {@link NeuralNetwork} object.
 */
public class Neuron implements Serializable {
	private final float[] weights;
    private float[] weightsChange;
    private final float[] squaredGradientsSum;
    private float[] gradientChange;
	private float value, bias, error, biasChange;
	
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
        weightsChange = new float[weightsSize];
        gradientChange = new float[weightsSize];
        squaredGradientsSum = new float[weightsSize];
	}
	
	/**
	 * Initialize all weights and bias with random float between -1 and 1
	 */

	public void initWeights() {
		Random random = new Random();
		bias = random.nextFloat(-1, 1);
		
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat(-1, 1);
	}
	
	/**
	 * Calculates value using equation: v = w * v + b of a {@link Layer} object,
	 * then applying it to the sigmoid function.
	 *
	 * @param layer {@link Layer} object containing neurons, used to calculate current neuron's value
	 */
	public void calcValue(Layer layer) {
		value = bias;
		error = 0;
		
		Neuron[] neurons = layer.getNeurons();
		for (int i = 0; i < neurons.length; i++)
			value += neurons[i].value * getWeight(i);
		
		value = sigmoid(value);
	}
	
	/**
	 * Calculates errors in a {@link Layer} object.
	 *
	 * @param layer layer to calculate errors for, used to calculate current neuron's error value.
	 */
	public void calcErrors(Layer layer) {
		Neuron[] neurons = layer.getNeurons();
		for (int i = 0; i < neurons.length; i++) {
			Neuron neuron = neurons[i];
			neuron.error += error * getWeight(i);
		}
	}
	
	/**
	 * Calculates the gradient of the cost function to find the global minimum.
	 *
	 * @param layer        {@link Layer} object to get neuron weights from
	 */
	public void calcWeightChange(Layer layer) {
		Neuron[] neurons = layer.getNeurons();
		for (int i = 0; i < neurons.length; i++) {
			Neuron neuron = neurons[i];
			float gradient = error * sigmoidDerivative(value);
            gradientChange[i] += gradient;
			weightsChange[i] += gradient * neuron.getValue();
		}
		biasChange += error;
	}

    public void applyWeightChange(float learningRate){
        for (int i = 0; i < getNumWeights(); i++) {
            float weightLearningRate = learningRate;
            squaredGradientsSum[i] = (float) ((0.9 * squaredGradientsSum[i]) + (0.1 * Math.pow(gradientChange[i], 2)));
            weightLearningRate /= (float) (Math.pow(squaredGradientsSum[i] + 1.0e-8, 0.5));

            addWeight(i, weightsChange[i] * weightLearningRate);
        }
        addBias(biasChange * learningRate);

        weightsChange = new float[weights.length];
        biasChange = 0;
        gradientChange = new float[weights.length];
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
	
	// region Getters/Setters
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
	
	public void setWeights(float[] weights) {
		if (weights.length != this.weights.length)
			throw new RuntimeException("Length of weights parameter does not match number of weights: " + weights.length + " != " + this.weights.length);
		
		for (int i = 0; i < getNumWeights(); i++)
			setWeight(i, weights[i]);
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
	
	public void addValue(float value) {
		this.value += value;
	}
	
	
	public float getBias() {
		return bias;
	}
	
	public void setBias(float bias) {
		this.bias = bias;
	}
	
	public void addBias(float bias) {
		this.bias += bias;
	}
	
	
	public float getError() {
		return error;
	}
	
	public void setError(float target) {
		this.error = target - value;
	}
	
	public void addError(float error) {
		this.error += error;
	}
	// endregion
}
