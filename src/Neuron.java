/**
 * A class that represents a value that can influence the next layer.
 */
public class Neuron {
	private final NeuralNetwork network;  // network node is located in
	private final Layer layer;  // layer node is located in
	private float[] weights;  // array of all weights from previous layer
	private float value;  // weights * value of weight sources
	
	public Neuron(Layer layer, NeuralNetwork network) {
		this.layer = layer;
		this.network = network;
		initWeight();
	}
	
	public Layer getLayer() {
		return layer;
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
	
	public void setValue(float value) {
		this.value = value;
	}
	
	public void addValue(float value) {
		this.value += value;
	}
	
	public NeuralNetwork getNetwork() {
		return network;
	}
	
	/**
	 * Initializes weights that node connects to in next node layer.
	 */
	private void initWeight() {
		int layerNum = layer.getLayerNum();
		if (layerNum >= network.getLayerLengths().length - 1)  // exit if last node layer (output layer)
			return;
		
		int size = network.getLayerLengths()[layerNum + 1];
		weights = new float[size];
	}
}
