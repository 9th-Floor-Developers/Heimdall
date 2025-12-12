import exceptions.FileNotDeleted;
import model.NeuralNetwork;
import model.data.NumberImage;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static utils.NumberUtils.getAllImgs;
import static utils.NumberUtils.getRandomImgs;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.DirectoryNotEmptyException;

class TrainerTest {
	private static Trainer trainer;
	private static int counter = 0;
	private static float[][] targets, inputs;
	private static int[] outputs;
	
	@BeforeAll
	static void initTrainer() throws Exception {
		NumberImage[] allImages = getAllImgs("./src/main/resources/numbers/");
		
		NumberImage[] images = getRandomImgs(allImages, 1000, 67);
		targets = new float[images.length][];
		inputs = new float[images.length][];
		outputs = new int[images.length];
		
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
	
	@AfterAll
	static void clearTestingData() throws DirectoryNotEmptyException, FileNotFoundException {
		File resultsDirectory = new File("./src/training-results/");
		File[] files = resultsDirectory.listFiles();
		//noinspection DataFlowIssue
		int length = files.length;
		
		for (int i = 0; i < counter; i++) {
			int idx = length + 1 - counter;
			File tempFile = new File(resultsDirectory.getPath() + "/" + idx);
			
			if (!tempFile.isDirectory())
				throw new FileNotFoundException(tempFile.getPath() + " Is Not A Directory");
			
			//noinspection DataFlowIssue
			for (File file : tempFile.listFiles())
				if (!file.delete())
					throw new FileNotDeleted(file.getPath() + " Failed To Delete");
			
			if (!tempFile.delete())
				throw new DirectoryNotEmptyException(tempFile.getPath() + " Could Not Be Deleted");
		}
	}
	
	@Test
	void IOAgent() throws Exception {
		File resultsDirectory = new File("./src/training-results/");
		File[] files = resultsDirectory.listFiles();
		//noinspection DataFlowIssue
		int length = files.length;
		
		//noinspection AssertWithSideEffects
		assert trainer.addLogger() != null;
		
		//noinspection DataFlowIssue
		assert resultsDirectory.listFiles().length == length + 1;
		
		trainer.saveAgent("testAgent.ser");
		counter++;
		assert trainer.loadAgent("./src/training-results/" + (length + 1) + "/testAgent.ser") != null;
	}
	
	@Test
	void regularTrain() throws Exception {
		NeuralNetwork[] agents = (NeuralNetwork[]) TestingUtils.getPrivate(trainer, "agents");
		
		int score = (int) TestingUtils.invokePrivate(
				trainer, "trainAgent",
				new Class[] {
						NeuralNetwork.class, float[][].class,
						float[][].class, int[].class, float.class
				},
				agents[0], inputs, targets, outputs, .01f
		);
		
		assert score >= 0;
	}
	
	@Test
	void getBestScore() {
		assert trainer.getBestScore() >= 0;
	}
}
