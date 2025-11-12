package model;

import java.util.Random;

/**
 * A class that represents a neural network and all the layers within.
 * <p>
 * Each {@link Layer} contains an array of {@link Neuron} objects.
 */
public class NeuralNetwork {
	private final Layer[] layers;
	private final int[] layerLengths;
	
	/**
	 * Creates a neural network and initializes all layers, neurons, and weights within
	 *
	 * @param layerLengths array containing number of {@link Neuron} objects in each {@link Layer},
	 *                     {@code layerLengths.length} should be number of layers in network.
	 */
	public NeuralNetwork(int[] layerLengths) {
		layers = new Layer[layerLengths.length];
		this.layerLengths = layerLengths;
		
		for (int i = 0; i < layerLengths.length; i++) {
			Layer layer = new Layer(i, this, layerLengths[i]);
			layers[i] = layer;
		}
	}
	
	/**
	 * Creates a neural network with already initialized {@link Layer} objects.
	 *
	 * @param layers array of already initialized {@link Layer} objects consisting of {@link Neuron} objects.
	 */
	public NeuralNetwork(Layer[] layers) {
		this.layers = layers;
		layerLengths = new int[layers.length];
		
		for (int i = 0; i < layers.length; i++)
			layerLengths[i] = layers[i].getNumNeurons();
	}
	
	/**
	 * Adjust all weights in a network randomly to return a new variation of the network
	 *
	 * @param scale how much the weights are changing (+ and - bounds for new random evolution)
	 * @return a new neural network with modified ("evolved") weights
	 * @see Layer
	 * @see Neuron
	 */
	public NeuralNetwork evolve(float scale) {
		NeuralNetwork newNetwork = this;
		Random random = new Random();
		
		for (int i = 1; i < layers.length; i++) {  // skip input layer
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				Neuron neuron = newNetwork.getNeuron(i, j);
				for (int k = 0; k < layers[i].getNeuron(j).getNumWeights(); k++) {
					float randFloat = random.nextFloat(-scale, scale);
					neuron.addWeight(k, randFloat);
				}
				neuron.addBias(random.nextFloat(-scale, scale));
			}
		}
		
		return newNetwork;
	}
	
	/**
	 * Returns network determined values of output layer.
	 * <p>
	 * When return value is compared with definitive answer array, the accuracy of the network can be determined.
	 *
	 * @param inputs values neural network is trained on
	 * @return values of output layer, should be used to compare definitive answer array.
	 * @see Layer
	 * @see Neuron
	 */
	public float[] calcOutputs(float[] inputs) {
		int outputLayerIdx = layers.length - 1;
		float[] outputs = new float[layers[outputLayerIdx].getNumNeurons()];
		
		for (int i = 0; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == 0) {
					neuron.setValue(inputs[j]);
					continue;
				}
				
				neuron.calcValue(layers[i - 1]);
				if (i == outputLayerIdx)
					outputs[j] = neuron.getValue();
			}
		}
		
		return outputs;
	}
	
	/**
	 * Apply back propagation process to neural network.
	 *
	 * @param target       desired output values
	 * @param learningRate difference to modify weights (0.0-0.5)
	 * @see Layer
	 * @see Neuron
	 */
	public float[] backProp(float[] target, float learningRate) {
        float[] outputError = new float[layers[layers.length - 1].getNeurons().length];

		for (int i = 1; i < layers.length; i++) {
			Neuron[] neurons = layers[i].getNeurons();
			for (int j = 0; j < neurons.length; j++) {
				Neuron neuron = getNeuron(i, j);
				if (i == layers.length - 1){
                    neuron.setError(target[j]);
                    outputError[j] = neuron.getError();
                }
				
				Layer layer = layers[i - 1];
				neuron.calcErrors(layer);
				neuron.calcWeightChange(learningRate, layer);
			}
		}

        return outputError;
	}

    public void applyWeights(int inputLength){
        for (int i = 0; i < layers.length; i++) {
            Neuron[] neurons = layers[i].getNeurons();
            for (int j = 0; j < neurons.length; j++) {
                Neuron neuron = getNeuron(i, j);
                neuron.applyWeightChange(inputLength);
            }
        }
    }
	
	// region Getters/Setters
	public Layer[] getLayers() {
		return layers;
	}
	
	public int[] getLayerLengths() {
		return layerLengths;
	}
	
	public Layer getLayer(int idx) {
		return layers[idx];
	}
	
	
	public Neuron getNeuron(int layer, int number) {
		return layers[layer].getNeuron(number);
	}
	// endregion
}
