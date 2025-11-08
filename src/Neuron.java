/**
 * A class that represents a single neuron.
 */
public class Neuron {
	private float[] weights;  // array of all weights for next layer
	private float value;  // weights * value of weight sources
	
	/**
	 * Initializes weights that node connects to in next node layer.
	 */
	public Neuron(int layerNum, NeuralNetwork network) {
		if (layerNum >= network.getLayerLengths().length - 1)  // exit if last node layer (output layer)
			return;
		
		int size = network.getLayerLengths()[layerNum + 1];
		weights = new float[size];
	}
	
	public float[] getWeights() {
		return weights;
	}
	
	public void printWeights() {
		if (weights == null)
			return;
		
		for (float weight : weights)
			System.out.println(weight);
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
	
	public void resetValue() {
		value = 0;
	}
	
	public void addValue(float value) {
		this.value += value;
	}
}
