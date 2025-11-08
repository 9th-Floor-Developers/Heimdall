package backprop;

import static numbers.NumberGenerator.getAllImgs;
import static numbers.NumberGenerator.getImg;

import numbers.NumberImage;
import training_data.BasicDataSets;

import java.util.Random;

public class Backprop {
	private static final Random random = new Random();
	
	public static void main(String[] args) throws Exception {
        /*
		NumberImage[] images = getAllImgs();
		float[][] targets = new float[images.length][];
		float[][] inputs = new float[images.length][];

		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i];
			
			float[] flatInputs = new float[image.pixels().length * image.pixels()[0].length];
			int idx = 0;
			for (int r = 0; r < image.pixels().length; r++) {
				for (int c = 0; c < image.pixels()[0].length; c++) {
					flatInputs[idx] = image.pixels()[r][c];
					idx++;
				}
			}
			inputs[i] = flatInputs;
			
			float[] target = new float[10];
			target[image.value()] = 1;
			targets[i] = target;
		}
		
		int data_amount = 1000;
		
		float[][] newInputs = new float[data_amount][];
		float[][] newTargets = new float[data_amount][];
		for (int i = 0; i < data_amount; i++) {
			Random random = new Random();
			int idx = random.nextInt(0, inputs.length);
			newInputs[i] = inputs[idx];
			newTargets[i] = targets[idx];
		}
		
		NeuralNetwork network = new NeuralNetwork(newInputs[0].length, 20, newTargets[0].length); // input, hidden, output
        */
		float[][] inputs = BasicDataSets.or_not6_inputs;
        int[] outputs = BasicDataSets.or_not6_outputs;
        int possibleOutputs = 2;
        float[][] targets = new float[outputs.length][];

        for (int i = 0; i < outputs.length; i++) {
            targets[i] = new float[possibleOutputs];
            targets[i][outputs[i]] = 1;
        }

        NeuralNetwork network = new NeuralNetwork(inputs[0].length, 2, targets[0].length);

		for (int generation = 1; generation <= 10000; generation++) {
			for (int i = 0; i < inputs.length; i++) {
				network.train(inputs[i], targets[i], .1f);
			}
			if (generation % 100 == 0)
				System.out.printf("Generation %d | Loss: %.6f%n", generation, network.totalLoss(inputs, targets));
		}
	}
}
