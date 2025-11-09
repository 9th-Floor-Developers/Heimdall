import java.util.ArrayList;
import java.util.Random;

/**
 * A class that represents a neural network and all the layers within.
 */
public class NeuralNetwork {
	private Layer[] layers;  // network consisting of layers
	private int[] layerLengths;  // length of all layers in network
	
	/**
	 * Creates a neural network and initializes all layers, neurons, and weights within
	 *
	 * @param layerLengths array containing number of layers and number of neurons in each layer, layerLengths.length should be number of layers
	 */
	public NeuralNetwork(int[] layerLengths) {
		layers = new Layer[layerLengths.length];
		this.layerLengths = layerLengths;
		
		for (int i = 0; i < layerLengths.length; i++) {
			Layer layer = new Layer(i, this, layerLengths[i]);
//			for (Neuron neuron : layer.getNeurons())
//				neuron.initWeights();
			layers[i] = layer;
		}
	}
	
	/**
	 * Creates a neural network with already initialized layers
	 *
	 * @param layers array of already initialized layers
	 */
	public NeuralNetwork(Layer[] layers) {
		this.layers = layers;
		layerLengths = new int[layers.length];
		
		for (int i = 0; i < layers.length; i++)
			layerLengths[i] = layers[i].getNumNeurons();
	}
	
	public Layer getLayer(int idx) {
		return layers[idx];
	}
	
	public Layer[] getLayers() {
		return layers;
	}
	
	public int[] getLayerLengths() {
		return layerLengths;
	}
	
	/**
	 * Adjust the weights randomly to make a new variation of the neural network
	 *
	 * @param scale how much the weights are changing (+ and - bounds for new random difference)
	 * @return new, updated neural network
	 */
	public NeuralNetwork evolve(float scale) {
		NeuralNetwork newNetwork = new NeuralNetwork(layerLengths);
		Random random = new Random();
		
		for (int i = 0; i < layers.length; i++) {
			if (i >= layers.length - 1)  // dont update weights in output layer
				continue;
			
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				for (int k = 0; k < layers[i].getNeuron(j).getNumWeights(); k++) {
					float randFloat = random.nextFloat(-scale, scale);
					newNetwork.getNeuron(i, j).addWeight(k, randFloat);
				}
			}
		}
		
		return newNetwork;
	}
	
	/**
	 * Returns values of output layer, used to determine definitive answer. Essentially the "run" function.
	 *
	 * @param inputs inputs of neural network
	 * @return values of output layer
	 */
	public ArrayList<Float> calculate(float[] inputs) {
		for (Layer layer : layers)
			for (Neuron neuron : layer.getNeurons())
				neuron.resetValue();  // initializes all neuron values to 0
		
		for (int i = 0; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				if (i == 0){
					getNeuron(i, j).setValue(inputs[j]);
				}
				else {
					Neuron neuron = getNeuron(i, j);
					for (int k = 0; k < neuron.getNumWeights(); k++) {
						neuron.addValue(getNeuron(i - 1, j).getValue() * neuron.getWeight(k));
						System.out.println(neuron.getValue() * neuron.getWeight(k));
					}
				}
			}
			
		}
		
		ArrayList<Float> outputs = new ArrayList<>();
		Layer outputLayer = layers[layers.length - 1];
		for (Neuron neuron : outputLayer.getNeurons())
			outputs.add(neuron.getValue());
		
		return outputs;
	}
	
	public Neuron getNeuron(int layer, int number) {
		return layers[layer].getNeuron(number);
	}
	
	public void LogWeights(){
		for (Layer layer : layers){
			for (Neuron neuron : layer.getNeurons()){
				for (float weight : neuron.getWeights()){
					System.out.println(weight);
				}
			}
		}
	}
}
