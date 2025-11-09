package backprop;

import static numbers.NumberUtils.getAllImgs;
import static numbers.NumberUtils.getImg;

import numbers.NumberImage;
import training_data.BasicDataSets;
import training_data.GeneratedDataSets;

import java.util.Random;

public class Backprop {
	private static final Random random = new Random();
	
	public static void main(String[] args) throws Exception {

		NumberImage[] images = getAllImgs("./src/numbers/dataset/");
		float[][] allTargets = new float[images.length][];
		float[][] allInputs = new float[images.length][];

		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i];

			allInputs[i] = image.to1D();
			allTargets[i] = image.toTarget();
		}
		
		int data_amount = 1000;
		
		float[][] inputs = new float[data_amount][];
		float[][] targets = new float[data_amount][];
		for (int i = 0; i < data_amount; i++) {
			Random random = new Random();
			int idx = random.nextInt(0, allInputs.length);
			inputs[i] = allInputs[idx];
			targets[i] = allTargets[idx];
		}
		
		NeuralNetwork network = new NeuralNetwork(inputs[0].length, 300, targets[0].length); // input, hidden, output

        /*
        GeneratedDataSets.generateAdder10DataSets(200);
		float[][] inputs = GeneratedDataSets.adder10_inputs;
        float[][] targets = GeneratedDataSets.adder10_targets;

        NeuralNetwork network = new NeuralNetwork(inputs[0].length, 100, targets[0].length);
         */

		for (int generation = 1; generation <= 20000; generation++) {
			for (int i = 0; i < inputs.length; i++) {
				network.train(inputs[i], targets[i], .004f);
			}
            System.out.printf("Generation %d | Loss: %.6f%n", generation, network.totalLoss(inputs, targets));
		}
	}
}
