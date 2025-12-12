package scripts.backprop;

import model.Layer;
import model.Neuron;

import java.util.Random;

class NeuralNetwork {
	int inputSize, hiddenSize, outputSize;
	float[][] weightsIH; // input→hidden
	float[][] weightsHO; // hidden→output
	float[] biasH, biasO;
	
	Random rand = new Random();
	
	NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;
		
		weightsIH = new float[hiddenSize][inputSize];
		weightsHO = new float[outputSize][hiddenSize];
		biasH = new float[hiddenSize];
		biasO = new float[outputSize];
		
		// random initialization
		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				weightsIH[i][j] = rand.nextFloat(-1, 1);
			}
			biasH[i] = rand.nextFloat(-1, 1);
		}
		
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize; j++) {
				weightsHO[i][j] = rand.nextFloat(-1, 1);
			}
			biasO[i] = rand.nextFloat(-1, 1);
		}
	}
	
	/**
	 * Adjust all weights in a network randomly to return a new variation of the current network.
	 *
	 * @param scale how much the weights are changing (+/- bounds for new random evolution),
	 *              should be between 0.0-0.5
	 * @return a clone of the current neural network but with slightly modified ("evolved") weights
	 * @throws CloneNotSupportedException if cloning of current network object fails
	 * @see Layer
	 * @see Neuron
	 */
	public model.NeuralNetwork evolve(float scale) throws CloneNotSupportedException {
		model.NeuralNetwork newNetwork = (model.NeuralNetwork) this.clone();
		Random random = new Random();
		
		for (int i = 1; i < layers.length; i++) {  // skip input layer
			for (int j = 0; j < layers[i].getNumNeurons(); j++) {
				Neuron newNeuron = getNeuron(i, j);
				for (int k = 0; k < newNeuron.getNumWeights(); k++)
					newNeuron.addWeight(k, random.nextFloat(-scale, scale));
				newNeuron.addBias(random.nextFloat(-scale, scale));
				newNetwork.setNeuron(i, j, newNeuron);
			}
		}
		
		return newNetwork;
	}
	
	float[] feedForward(float[] input) {
		float[] hidden = new float[hiddenSize];
		for (int i = 0; i < hiddenSize; i++) {
			float sum = biasH[i];
			for (int j = 0; j < inputSize; j++) {
				sum += weightsIH[i][j] * input[j];
			}
			hidden[i] = sigmoid(sum);
		}
		
		float[] output = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			float sum = biasO[i];
			for (int j = 0; j < hiddenSize; j++) {
				sum += weightsHO[i][j] * hidden[j];
			}
			output[i] = sigmoid(sum);
		}
		return output;
	}
	
	void train(float[] input, float[] target, float learningRate) {
		// --- Forward ---
		float[] hidden = new float[hiddenSize];
		for (int i = 0; i < hiddenSize; i++) {
			float sum = biasH[i];
			for (int j = 0; j < inputSize; j++) {
				sum += weightsIH[i][j] * input[j];
			}
			hidden[i] = sigmoid(sum);
		}
		
		float[] outputs = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			float sum = biasO[i];
			for (int j = 0; j < hiddenSize; j++) {
				sum += weightsHO[i][j] * hidden[j];
			}
			outputs[i] = sigmoid(sum);
		}
		
		// --- Backward ---
		float[] outputErrors = new float[outputSize];
		for (int i = 0; i < outputSize; i++) {
			outputErrors[i] = target[i] - outputs[i];
		}
		
		float[] hiddenErrors = new float[hiddenSize];
		for (int i = 0; i < hiddenSize; i++) {
			float error = 0;
			for (int j = 0; j < outputSize; j++) {
				error += outputErrors[j] * weightsHO[j][i];
			}
			hiddenErrors[i] = error;
		}
		
		// Update output weights
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize; j++) {
				float gradient = outputErrors[i] * sigmoidDerivative(outputs[i]);
				weightsHO[i][j] += learningRate * gradient * hidden[j];
			}
			biasO[i] += learningRate * outputErrors[i];
		}
		
		// Update input→hidden weights
		for (int i = 0; i < hiddenSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				float gradient = hiddenErrors[i] * sigmoidDerivative(hidden[i]);
				weightsIH[i][j] += learningRate * gradient * input[j];
			}
			biasH[i] += learningRate * hiddenErrors[i];
		}
	}
	
	float totalLoss(float[][] inputs, float[][] targets) {
		float total = 0;
		for (int i = 0; i < inputs.length; i++) {
			float[] out = feedForward(inputs[i]);
			for (int j = 0; j < out.length; j++) {
				total += Math.pow(targets[i][j] - out[j], 2);
			}
		}
		return total / inputs.length;
	}
	
	float sigmoid(float x) {
		return (float) (1 / (1 + Math.exp(-x)));
	}
	
	float sigmoidDerivative(float y) {
		return y * (1 - y);
	}
}
