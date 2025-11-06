import java.util.ArrayList;
import java.util.Random;

/**
 * A class that represents a neural network and all the nodes within.
 */
public class NeuralNetwork {
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
		
//		nodes.forEach(layer -> layer.forEach(Node::initWeight));  // initializing weights for all nodes
	}
	
	public ArrayList<ArrayList<Node>> getNodes() {
		return nodes;
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
		
		for (int i = 0; i < nodes.size(); i++) {
			if (i >= nodes.size() - 1)  // dont update weights in output layer
				continue;
			
			for (int j = 0; j < nodes.get(i).size(); j++) {
				for (int k = 0; k < nodes.get(i).get(j).getWeights().length; k++) {
					float randFloat = random.nextFloat(-scale, scale);  // new random number based on scale
					// updating every weight of every node with new random value
//					newNetwork.nodes.get(i).get(j).getWeights()[k] = nodes.get(i).get(j).getWeights()[k] + randFloat;
					newNetwork.getNode(i, j).addWeight(k, randFloat);;
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
		nodes.forEach(layer -> layer.forEach(n -> n.setValue(0)));  // initializes all values at 0
		
		for (int i = 0; i < nodes.size(); i++) {
			if (i == 0) {  // checks if first layer
				for (int j = 0; j < inputs.length; j++) {
					getNode(0, j).addValue(inputs[j]);
//					nodes.getFirst().get(j).addValue(inputs[j]);  // sets first layer values as input values
				}
			}
			
			if (i >= nodes.size() - 1)
				continue;
			
			for (Node node : nodes.get(i)) {
				for (int j = 0; j < node.getWeights().length; j++) {
//					Node node1 = getNode(i + 1, j); // nodes.get(i + 1).get(j);
					// prev node value * prev node weight
					// repeat for all nodes in prev layer
					getNode(i + 1, j).addValue(node.getValue() * node.getWeights()[j]);
				}
			}
			
		}
		
		return new ArrayList<>(nodes.getLast().stream().map(Node::getValue).toList());
	}
	
	private Node getNode(int layer, int number) {
		return nodes.get(layer).get(number);
	}

    public ArrayList<Node> getAllNodes(){
        ArrayList<Node> allNodes = new ArrayList<>();
        nodes.forEach(allNodes::addAll);
        return allNodes;
    }

    public void logAllWeights(){
        for (Node node : getAllNodes()){
            node.logWeights();
        }
    }
}
