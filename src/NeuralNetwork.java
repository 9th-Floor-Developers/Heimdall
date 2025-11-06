import java.util.ArrayList;
import java.util.Random;

/**
 * A class that represents a neural network and all the neurons within.
 */
public class NeuralNetwork {
	private final ArrayList<ArrayList<Neuron>> neurons;  // network consisting of neurons
	private final int[] layers;  // array of all layers in the network
	
	public NeuralNetwork(int[] layers) {
		this.layers = layers;
		
		neurons = new ArrayList<>();
		for (int i = 0; i < layers.length; i++) {
			ArrayList<Neuron> layer = new ArrayList<>();
			
			for (int j = 0; j < layers[i]; j++)
				layer.add(new Neuron(i, this));  // initializing all neurons with layer number and network
			
			neurons.add(layer);
		}
	}
	
	public ArrayList<ArrayList<Neuron>> getNeurons() {
		return neurons;
	}
	
	public int[] getLayers() {
		return layers;
	}
	
	/**
	 * Adjust the weights randomly to make a new variation of the neural network
	 *
	 * @param scale how much the weights are changing (+ and - bounds for new random difference)
	 * @return new, updated neural network
	 */
	public NeuralNetwork evolve(float scale) {
		NeuralNetwork newNetwork = new NeuralNetwork(layers);
		Random random = new Random();
		
		for (int i = 0; i < neurons.size(); i++) {
			if (i >= neurons.size() - 1)  // dont update weights in output layer
				continue;
			
			for (int j = 0; j < neurons.get(i).size(); j++) {
				for (int k = 0; k < neurons.get(i).get(j).getWeights().length; k++) {
					float randFloat = random.nextFloat(-scale, scale);  // new random number based on scale
					newNetwork.getNode(i, j).addWeight(k, randFloat);
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
		neurons.forEach(layer -> layer.forEach(n -> n.setValue(0)));  // initializes all values at 0
		
		for (int i = 0; i < neurons.size(); i++) {
			if (i == 0) {  // checks if first layer
				for (int j = 0; j < inputs.length; j++) {
					getNode(0, j).addValue(inputs[j]);
				}
			}
			
			if (i >= neurons.size() - 1)
				continue;
			
			for (Neuron neuron : neurons.get(i)) {
				for (int j = 0; j < neuron.getWeights().length; j++) {
					// prev node value * prev node weight
					// repeat for all neurons in prev layer
					getNode(i + 1, j).addValue(neuron.getValue() * neuron.getWeights()[j]);
				}
			}
			
		}
		
		return new ArrayList<>(neurons.getLast().stream().map(Neuron::getValue).toList());
	}
	
	public Neuron getNode(int layer, int number) {
		return neurons.get(layer).get(number);
	}
	
	public ArrayList<Neuron> getAllNodes() {
		ArrayList<Neuron> allNeurons = new ArrayList<>();
		neurons.forEach(allNeurons::addAll);
		return allNeurons;
	}
	
	public void printAllWeights() {
		for (Neuron neuron : getAllNodes()) {
			neuron.printWeights();
		}
	}
}
