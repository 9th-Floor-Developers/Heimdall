import java.util.Arrays;
import java.util.Random;

/**
 * A class that represents a single neuron.
 */
public class Neuron {
	private float[] weights;  // array of all weights for next layer
	private float value;  // weights * value of weight sources
	
	public Neuron(int weightsSize) {
		weights = new float[weightsSize];
	}
	
	public void initWeights() {
		Random random = new Random();
		for (int i = 0; i < weights.length; i++)
			weights[i] = random.nextFloat(-1, 1);
	}
	
//	public Neuron(int layerNum, NeuralNetwork network) {
//		if (layerNum == 0) {  // exit if last node layer (output layer)
//			return;
//		}
//
//		int size = network.getLayerLengths()[layerNum - 1];
//		weights = new float[size];
//	}
	
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
		this.weights = weights;
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
}
