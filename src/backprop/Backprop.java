package backprop;

import static numbers.NumberGenerator.getAllImgs;
import numbers.NumberImage;

import java.util.Random;

public class Backprop {
	private static final Random random = new Random();
	
	public static void main(String[] args) throws Exception {
		NumberImage[] images = getAllImgs();
		
//		float[][] allInputs = new float[images.length][];
//		int[] allOutputs = new int[images.length];
//		for (int i = 0; i < images.length; i++) {
//			NumberImage image = images[i];
//			float[] flatInputs = new float[image.pixels().length * image.pixels()[0].length];
//			int idx = 0;
//			for (int r = 0; r < image.pixels().length; r++) {
//				for (int c = 0; c < image.pixels()[0].length; c++) {
//					flatInputs[idx] = image.pixels()[r][c];
//					idx++;
//				}
//			}
//
//			allInputs[i] = flatInputs;
//			allOutputs[i] = image.value();
//		}
//
//		int data_amount = 2000;
//		float[][] inputs = new float[data_amount][];
//		int[] outputs = new int[data_amount];
//		for (int i = 0; i < data_amount; i++) {
//			Random random = new Random();
//			int idx = random.nextInt(0, allInputs.length);
//			inputs[i] = allInputs[idx];
//			outputs[i] = allOutputs[idx];
//		}
		
		// OR gate dataset
		
		
		
		
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
		
		
		for (int generation = 1; generation <= 10000; generation++) {
			for (int i = 0; i < data_amount; i++) {
				network.train(newInputs[i], newTargets[i], .1f);
			}
			if (generation % 100 == 0)
				System.out.printf("Generation %d | Loss: %.6f%n", generation, network.totalLoss(newInputs, newTargets));
		}
	}
}
