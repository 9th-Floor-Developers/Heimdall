import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A class that represents a neural network and all the nodes within.
 */
public class NeuralNetwork {
	/**
	 * A class that represents a value that can influence the next layer.
	 */
	private static class Node {
		private final int layer;  // layer node is located in
		private float[] weights;  // array of all weights from previous layer
		private float value;  // weights * value of weight sources
		private final NeuralNetwork network;  // network node is located in
		
		private Node(int layer, NeuralNetwork network) {
			this.layer = layer;
			this.network = network;
		}
		
		/**
		 * Initializes weights that node connects to in next node layer.
		 */
		private void initWeight() {
			if (layer >= network.nodes.size() - 1)  // exit if last node layer
				return;
			
			int size = network.nodes.get(layer + 1).size();
			weights = new float[size];
			for (int i = 0; i < size; i++)
				weights[i] = 0;  // setting all values to 0 (default)
			
		}
	}
	
	private final ArrayList<ArrayList<Node>> nodes;  // network consisting of nodes
	private final int[] layers;  // array of all layers in the network
	
	public NeuralNetwork(int[] layers) {
		this.layers = layers;
		
		nodes = new ArrayList<>();
		for (int i = 0; i < layers.length; i++) {
			ArrayList<Node> layer = new ArrayList<>();
			
			for (int j = 0; j < layers[i]; j++)
				layer.add(new Node(i, this));  // initializing all nodes with layer number and network
			
			nodes.add(layer);
		}
		
		nodes.forEach(layer -> layer.forEach(Node::initWeight));  // initializing weights for all nodes
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
		
		for (int i = 0; i < nodes.size(); i++) {
			if (i >= nodes.size() - 1)  // dont update weights in output layer
				continue;
			
			for (int j = 0; j < nodes.get(i).size(); j++) {
				for (int k = 0; k < nodes.get(i).get(j).weights.length; k++) {
					float randFloat = random.nextFloat(-scale, scale);  // new random number based on scale
					// updating every weight of every node with new random value
					newNetwork.nodes.get(i).get(j).weights[k] = nodes.get(i).get(j).weights[k] + randFloat;
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
		nodes.forEach(layer -> layer.forEach(n -> n.value = 0));  // initializes all values at 0
		
		for (int i = 0; i < nodes.size(); i++) {
			if (i == 0)  // checks if first layer
				for (int j = 0; j < inputs.length; j++)
					nodes.getFirst().get(j).value += inputs[j];  // sets first layer values as input values
			
			if (i >= nodes.size() - 1)
				continue;
			
			for (Node node : nodes.get(i))
				for (int j = 0; j < node.weights.length; j++)
					// prev node value * prev node weight
					// repeat for all nodes in prev layer
					nodes.get(i + 1).get(j).value += node.value * node.weights[j];
			
		}
		
		return new ArrayList<>(nodes.getLast().stream().map(n -> n.value).toList());
	}
}
