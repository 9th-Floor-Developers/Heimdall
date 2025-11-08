import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class that represents a single layer containing nodes.
 */
public class Layer {
	private final Neuron[] neurons;  // all neurons in layer
	
	/**
	 * Initialize nodes in current layer based on parameters
	 *
	 * @param layerNum   index of this layer in neural network
	 * @param network    neural network this layer is located in
	 * @param numNeurons number of neurons to include in this layer
	 */
	public Layer(int layerNum, NeuralNetwork network, int numNeurons) {
		neurons = new Neuron[numNeurons];
		
		for (int i = 0; i < numNeurons; i++)
			neurons[i] = new Neuron(layerNum, network);
	}
	
	public Neuron getNeuron(int idx) {
		return neurons[idx];
	}
	
	public Neuron[] getNeurons() {
		return neurons;
	}
	
	public ArrayList<Neuron> getNeuronsList() {
		return new ArrayList<>(Arrays.asList(neurons));
	}
	
	public int getNumNeurons() {
		return neurons.length;
	}
}
