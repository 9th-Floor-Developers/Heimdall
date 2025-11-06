import shapes.Shape;
import shapes.ShapeGenerator;

public class Heimdall {
	public static void main(String[] args) {
		BasicTrainer trainer = new BasicTrainer();
		trainer.train(
            BasicDataSet.or_not3_inputs,
            BasicDataSet.or_not3_outputs,
            new int[]{3, 2, 2},
            5,
            5
        );
	}
}
