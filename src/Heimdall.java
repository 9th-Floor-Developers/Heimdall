import shapes.Shape;
import shapes.ShapeGenerator;

public class Heimdall {
	public static void main(String[] args) {
		BasicTrainer trainer = new BasicTrainer();
		trainer.train(
            BasicDataSet.or2_inputs,
            BasicDataSet.or2_outputs,
            new int[]{2, 2}
        );
	}
}
