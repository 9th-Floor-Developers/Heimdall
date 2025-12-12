import model.Layer;
import model.NeuralNetwork;
import model.Neuron;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

class NeuralNetworkTest {
	private static NeuralNetwork network;
	
	@BeforeAll
	static void setUp() {
		int[] layerLengths = {
				500,
				100,
				10
		};
		
		network = new NeuralNetwork(layerLengths, 123);
	}
	
	@Test
	void initialValues() {
		// checks if all initial values are in range: -1 < v < 1
		for (float[][] layerWeights : network.getWeights())
			for (float[] neuronWeights : layerWeights)
				for (float weight : neuronWeights)
					assert weight > -1 && weight < 1;
	}
	
	@Test
	void similarGettersSetters() throws Exception {
		Layer[] layers = (Layer[]) TestingUtils.getPrivate(network, "layers");
		int iLength = layers.length - 1;
		Neuron[][] neurons = new Neuron[layers.length][];
		
		float[][][] weights = new float[iLength][][];
		float[][] biases = new float[iLength][],
				values = new float[iLength][],
				errors = new float[iLength][];
		
		for (int i = 0; i < layers.length; i++)
			neurons[i] = (Neuron[]) TestingUtils.getPrivate(layers[i], "neurons");
		
		for (int i = 0; i < neurons.length - 1; i++) {
			int jLength = neurons[i + 1].length;
			
			weights[i] = new float[jLength][];
			biases[i] = new float[jLength];
			values[i] = new float[jLength];
			errors[i] = new float[jLength];
			
			for (int j = 0; j < jLength; j++) {
				Neuron target = neurons[i + 1][j];
				
				weights[i][j] = (float[]) TestingUtils.getPrivate(target, "weights");
				biases[i][j] = (float) TestingUtils.getPrivate(target, "bias");
				values[i][j] = (float) TestingUtils.getPrivate(target, "value");
				errors[i][j] = (float) TestingUtils.getPrivate(target, "error");
			}
		}
		
		int[] layerLengths = new int[layers.length];
		for (int i = 0; i < layers.length; i++)
			layerLengths[i] = neurons[i].length;
		
		assert Arrays.deepEquals(layers, network.getLayers()) && layers[0] == network.getLayer(0);
		assert Arrays.equals(layerLengths, network.getLayerLengths());
		assert Arrays.deepEquals(neurons, network.getNeurons()) && neurons[0][0] == network.getNeuron(0, 0);
		assert Arrays.deepEquals(weights, network.getWeights());
		assert Arrays.deepEquals(biases, network.getBiases());
		assert Arrays.deepEquals(values, network.getValues());
		assert Arrays.deepEquals(errors, network.getErrors());
		
		
		float[][][] newWeights = Arrays.stream(weights)
		                               .map(layer -> Arrays.stream(layer)
		                                                   .map(float[]::clone)
		                                                   .toArray(float[][]::new))
		                               .toArray(float[][][]::new);
		
		float[][] newBiases = Arrays.stream(biases)
		                            .map(float[]::clone)
		                            .toArray(float[][]::new),
				newValues = Arrays.stream(values)
				                  .map(float[]::clone)
				                  .toArray(float[][]::new),
				newErrors = Arrays.stream(errors)
				                  .map(float[]::clone)
				                  .toArray(float[][]::new);
		
		newWeights[0][0][0] += 1.111f;
		newBiases[0][0] += 2.222f;
		newValues[0][0] += 3.333f;
		newErrors[0][0] += 4.444f;
		
		network.setWeights(newWeights);
		network.setBiases(newBiases);
		network.setValues(newValues);
		network.setErrors(newErrors);
		
		assert Arrays.deepEquals(newWeights, network.getWeights());
		assert Arrays.deepEquals(newBiases, network.getBiases());
		assert Arrays.deepEquals(newValues, network.getValues());
		assert Arrays.deepEquals(newErrors, network.getErrors());
		
		Neuron[][] newNeurons = new Neuron[neurons.length][];
		for (int i = 0; i < neurons.length; i++)
			newNeurons[i] = neurons[i].clone();
		
		newNeurons[1][0] = new Neuron(network.getLayerLengths()[0] + 1);
		network.setNeurons(newNeurons);
		
		assert Arrays.deepEquals(newNeurons, network.getNeurons());
		
		Neuron replacement = new Neuron(5);
		network.setNeuron(1, 0, replacement);
		assert network.getNeuron(1, 0) == replacement;
		
		Layer newLayer = new Layer(0, network, 99, new Random(123));
		network.setLayer(0, newLayer);
		assert network.getLayer(0) == newLayer;
	}
	
	@Test
	void calcOutputs() {
	}
	
	@Test
	void backProp() {
	}
	
	@Test
	void applyWeights() {
	}
	
	@Test
	void getLayers() {
	}
	
	@Test
	void getLayerLengths() {
	}
	
	@Test
	void getLayer() {
	}
	
	@Test
	void getNeuron() {
	}
	
	@Test
	void setNeuron() {
	}
	
	@Test
	void setWeights() {
	}
	
	@Test
	void getWeights() {
	}
	
	@Test
	void setBiases() {
	}
	
	@Test
	void getBiases() {
	}
}
