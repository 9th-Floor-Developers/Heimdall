import java.util.ArrayList;
import java.util.Random;

/**
 * A class that represents a neural network and all the layers within.
 */
public class NeuralNetwork {
	private final Layer[] layers;  // network consisting of layers
	private final int[] layerLengths;  // length of all layers in network
	
	public NeuralNetwork(int[] layerLengths) {
		this.layerLengths = layerLengths;
		
		layers = new Layer[layerLengths.length];
		for (int i = 0; i < layerLengths.length; i++) {
			Layer layer = new Layer(i, this, layerLengths[i]);
			layers[i] = layer;
		}
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
				for (int k = 0; k < layers[i].getNeuron(j).getWeights().length; k++) {
					float randFloat = random.nextFloat(-scale, scale);  // new random number based on scale
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
		
		for (int i = 0; i < layers.length - 1; i++) {
			if (i == 0)  // checks if first layer
				for (int j = 0; j < inputs.length; j++)
					getNeuron(0, j).addValue(inputs[j]);
			
			for (Neuron neuron : layers[i].getNeurons()) {
				for (int j = 0; j < neuron.getWeights().length; j++) {
					// prev node value * prev node weight
					// repeat for all nodes in next layer
					getNeuron(i + 1, j).addValue(neuron.getValue() * neuron.getWeights()[j]);
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
}
