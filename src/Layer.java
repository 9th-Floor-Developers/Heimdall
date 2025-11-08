import java.util.ArrayList;
import java.util.Arrays;

public class Layer {
	private final int layerNum;
	private final Neuron[] neurons;
	private final NeuralNetwork network;
	
	public Layer(int layerNum, NeuralNetwork network, int numNeurons) {
		this.layerNum = layerNum;
		this.network = network;
		neurons = new Neuron[numNeurons];
		
		for (int i = 0; i < numNeurons; i++)
			neurons[i] = new Neuron(this, network);
	}
	
	public int getLayerNum() {
		return layerNum;
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
	
	public NeuralNetwork getNetwork() {
		return network;
	}
}
