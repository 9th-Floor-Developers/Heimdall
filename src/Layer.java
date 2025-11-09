import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class that represents a single layer containing nodes.
 */
public class Layer {
	private final Neuron[] neurons;  // all neurons in layer
	
	public Layer(int numNeurons) {
		neurons = new Neuron[numNeurons];
	}
	
	public Layer(int layerNum, NeuralNetwork network, int numNeurons) {
		neurons = new Neuron[numNeurons];
		
		for (int i = 0; i < numNeurons; i++) {
			if (layerNum == 0) {
				Neuron neuron = new Neuron(0);
				neuron.initWeights();  // TODO: remove
				neurons[i] = neuron;
				continue;
			}
			
			int numWeights = network.getLayer(layerNum - 1).getNumNeurons();
			Neuron neuron = new Neuron(numWeights);
			neuron.initWeights();
			neurons[i] = neuron;
		}
	}
	
	public Neuron getNeuron(int idx) {
		return neurons[idx];
	}
	
	public void setNeuron(int idx, Neuron neuron) {
		neurons[idx] = neuron;
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
