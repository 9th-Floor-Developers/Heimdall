package scripts.backprop;

import model.data.NumberImage;
import static utils.NumberUtils.getRandomImgs;

import java.text.DecimalFormat;

public class Backprop {
	public static void main(String[] args) throws Exception {
		NumberImage[] images = getRandomImgs("./src/datasets/numbers/", 1000);
		float[][] targets = new float[images.length][],
				inputs = new float[images.length][];
		
		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i].scaleDownImage(2);
			inputs[i] = image.to1D();
			targets[i] = image.toTarget();
		}
		
		System.out.println("Training...");
		
		NeuralNetwork network = new NeuralNetwork(
				inputs[0].length,  // input
				300,  // hidden layer
				targets[0].length  // output
		);
		
		for (int generation = 1; generation <= 20000; generation++) {
			for (int i = 0; i < inputs.length; i++)
				network.train(inputs[i], targets[i], .004f);
			float loss = network.totalLoss(inputs, targets) * 100;
			String formatted = new DecimalFormat("###.##").format(loss);
			System.out.println("Generation " + generation + " | Loss: " + formatted + "%");
			
			if (loss <= 5)
				break;
		}
	}
}
