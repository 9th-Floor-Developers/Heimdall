import static datasets.BasicDataSets.*;
import datasets.GeneratedDataSets;
import model.data.NumberImage;
import static utils.NumberUtils.getAllImgs;
import static utils.NumberUtils.getRandomImgs;

import java.text.DecimalFormat;

public class Heimdall {
	/**
	 * Entry point
	 */
	public static void main(String[] args) throws Exception {
		float[][] inputs = new float[][]{
				new float[]{1, 0}, new float[]{0, 1}, new float[]{1, 1}, new float[]{0, 0}
		};
		float[][] targets = new float[][]{
				new float[]{1, 0}, new float[]{1, 0}, new float[]{1, 0}, new float[]{0, 1}
		};
		int[] outputs = new int[]{
				0, 0, 0, 1
		};
		
		Trainer trainer = new Trainer(
				1,  // number of agents per round, more possibilities to evolve
				new int[] {  // layers format
						inputs[0].length,  // input layer must match input count // number of middle layer nodes, more opportunities per agent to learn
						targets[0].length  // output layer is number of possible answers (0.0-1.0 inclusive)
				}
		);
		
		for (int generation = 1; generation <= 1000; generation++) {
			float loss = trainer.train(
					inputs,
					targets,
					outputs,
					.5f
			);
			
			String formatted = new DecimalFormat("###.##").format(loss);
			System.out.println("Generation " + generation + " | Loss: " + formatted + "%");
			
//			if (loss <= 5)
//				break;
		}
		
	}
}
