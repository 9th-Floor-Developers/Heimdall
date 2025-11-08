package training_data;

import java.util.Random;

public class GeneratedDataSets {
    public static float[][] adder10_inputs;
    public static float[][] adder10_targets;

    public static void generateAdder10DataSets(int dataAmount){
        Random random = new Random();

        adder10_inputs = new float[dataAmount][];
        adder10_targets = new float[dataAmount][];

        for (int i = 0; i < dataAmount; i++){
            float[] input = new float[10];

            int output = 0;
            for (int j = 0; j < 10; j++){
                input[j] = (float) random.nextInt(2);
                if (input[j] == 1){
                    output++;
                }
            }

            float[] targets = new float[11];
            targets[output] = 1;

            adder10_inputs[i] = input;
            adder10_targets[i] = targets;
        }
    }
}
