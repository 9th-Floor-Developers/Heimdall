import model.data.NumberImage;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static utils.NumberUtils.getAllImgs;
import static utils.NumberUtils.getRandomImgs;

import java.io.File;

class TrainerTest {
	private static Trainer trainer;
	
	@BeforeAll
	public static void initTrainer() throws Exception {
		NumberImage[] allImages = getAllImgs("./src/datasets/numbers/");
		
		NumberImage[] images = getRandomImgs(allImages, 1000, 67);
		float[][] targets = new float[images.length][],
				inputs = new float[images.length][];
		int[] outputs = new int[images.length];
		
		for (int i = 0; i < images.length; i++) {
			NumberImage image = images[i];
			inputs[i] = image.to1D();
			targets[i] = image.toTarget();
			outputs[i] = image.value();
		}
		
		trainer = new Trainer(
				10,
				new int[] {
						inputs[0].length,
						100,
						targets[0].length
				}
		);
	}
	
	@Test
	public void IOAgent() throws Exception {
		File resultsDirectory = new File("./src/training-results/");
		File[] files = resultsDirectory.listFiles();
		//noinspection DataFlowIssue
		int length = files.length;
		
		//noinspection AssertWithSideEffects
		assert trainer.addLogger() != null;
		
		//noinspection DataFlowIssue
		assert resultsDirectory.listFiles().length == length + 1;
		
		trainer.saveAgent("testAgent.ser");
		assert trainer.loadAgent("./src/training-results/" + (length + 1) + "/testAgent.ser") != null;
	}
	
	@Test
	public void regularTrain() {
	}
	
	@Test
	public void evolutionTrain() {
	}
	
	@Test
	public void getBestScore() {
	}
}
