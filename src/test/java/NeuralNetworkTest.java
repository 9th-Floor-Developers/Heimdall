import model.Layer;
import model.NeuralNetwork;
import model.Neuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

class NeuralNetworkTest {
	private NeuralNetwork network;
	private Layer[] layers;
	
	@BeforeEach
	void setUp() throws Exception {
		int[] layerLengths = {
				2,
				2,
				10
		};
		
		network = new NeuralNetwork(layerLengths, 123);
		layers = (Layer[]) TestingUtils.getPrivate(network, "layers");
	}
	
	@Test
	void initialValues() {
		// checks if all initial values are in range: -1 < v < 1
        checkWeights();
    }

    private void checkWeights() {
        for (float[][] layerWeights : network.getWeights())
            for (float[] neuronWeights : layerWeights)
                for (float weight : neuronWeights)
                    assert weight > -1 && weight < 1;
    }

    @Test
	void calcOutputs() {
        network.calcOutputs(new float[]{1, 0});

        Layer layer = network.getLayer(network.getLayers().length - 1);
        for (Neuron neuron : layer.getNeurons())
            assert neuron.getValue() > -1 && neuron.getValue() < 1;
	}
	
	@Test
	void backProp() {
        checkWeights();
    }
	
	@Test
	void applyWeightsChange() {
        checkWeights();
    }
	
	@Test
	void layers() {
		assert Arrays.deepEquals(layers, network.getLayers()) && layers[0] == network.getLayer(0);
		
		Layer newLayer = new Layer(0, network, 99, new Random(123));
		network.setLayer(0, newLayer);
		assert network.getLayer(0) == newLayer;
	}
	
	@Test
	void layerLengths() {
		int[] layerLengths = new int[layers.length];
		for (int i = 0; i < layers.length; i++)
			layerLengths[i] = layers[i].getNumNeurons();
		assert Arrays.equals(layerLengths, network.getLayerLengths());
	}
	
	@Test
	void neurons() throws Exception {
		Neuron[][] neurons = new Neuron[layers.length][];
		
		for (int i = 0; i < layers.length; i++)
			neurons[i] = (Neuron[]) TestingUtils.getPrivate(layers[i], "neurons");
		
		assert Arrays.deepEquals(neurons, network.getNeurons()) && neurons[0][0] == network.getNeuron(0, 0);
		
		Neuron[][] newNeurons = new Neuron[neurons.length][];
		for (int i = 0; i < neurons.length; i++)
			newNeurons[i] = neurons[i].clone();
		
		newNeurons[1][0] = new Neuron(network.getLayerLengths()[0] + 1);
		network.setNeurons(newNeurons);
		
		assert Arrays.deepEquals(newNeurons, network.getNeurons());
		
		Neuron replacement = new Neuron(5);
		network.setNeuron(1, 0, replacement);
		assert network.getNeuron(1, 0) == replacement;
	}
	
	@Test
	void weights() throws Exception {
		float[][][] weights = new float[layers.length - 1][][];
		
		for (int i = 0; i < layers.length - 1; i++) {
			Layer layer = layers[i + 1];
			int numNeurons = layer.getNumNeurons();
			
			weights[i] = new float[numNeurons][];
			for (int j = 0; j < numNeurons; j++)
				weights[i][j] = (float[]) TestingUtils.getPrivate(layer.getNeuron(j), "weights");
		}
		
		assert Arrays.deepEquals(weights, network.getWeights());
		
		weights[0][0][0] += 1.234f;
		network.setWeights(weights);
		assert Arrays.deepEquals(weights, network.getWeights());
	}
	
	@Test
	void biases() throws Exception {
		float[][] biases = new float[layers.length - 1][];
		
		for (int i = 0; i < layers.length - 1; i++) {
			Layer layer = layers[i + 1];
			int numNeurons = layer.getNumNeurons();
			
			biases[i] = new float[numNeurons];
			for (int j = 0; j < numNeurons; j++)
				biases[i][j] = (float) TestingUtils.getPrivate(layer.getNeuron(j), "bias");
		}
		
		assert Arrays.deepEquals(biases, network.getBiases());
		
		biases[0][0] += 1.234f;
		network.setBiases(biases);
		assert Arrays.deepEquals(biases, network.getBiases());
	}
	
	@Test
	void values() throws Exception {
		float[][] values = new float[layers.length - 1][];
		
		for (int i = 0; i < layers.length - 1; i++) {
			Layer layer = layers[i + 1];
			int numNeurons = layer.getNumNeurons();
			
			values[i] = new float[numNeurons];
			for (int j = 0; j < numNeurons; j++)
				values[i][j] = (float) TestingUtils.getPrivate(layer.getNeuron(j), "value");
		}
		
		assert Arrays.deepEquals(values, network.getValues());
		
		values[0][0] += 1.234f;
		network.setValues(values);
		assert Arrays.deepEquals(values, network.getValues());
	}
	
	@Test
	void errors() throws Exception {
		float[][] errors = new float[layers.length - 1][];
		
		for (int i = 0; i < layers.length - 1; i++) {
			Layer layer = layers[i + 1];
			int numNeurons = layer.getNumNeurons();
			
			errors[i] = new float[numNeurons];
			for (int j = 0; j < numNeurons; j++)
				errors[i][j] = (float) TestingUtils.getPrivate(layer.getNeuron(j), "error");
		}
		
		assert Arrays.deepEquals(errors, network.getErrors());
		
		errors[0][0] += 1.234f;
		network.setErrors(errors);
		assert Arrays.deepEquals(errors, network.getErrors());
	}
}
