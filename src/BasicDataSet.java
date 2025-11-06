public class BasicDataSet {
    public static final float[][] or5_inputs = new float[][] {
            {0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0},
            {1, 0, 0, 0, 1},
            {1, 0, 1, 0, 1},
            {1, 1, 1, 1, 1}
    };

    public static final int[] or5_outputs = new int[] {
            0, 1, 1, 1, 1
    };

    public static final float[][] or2_inputs = new float[][] {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
    };

    public static final int[] or2_outputs = new int[] {
            0, 1, 1, 1
    };

    public static final float[][] or_not3_inputs = new float[][] {
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {0, 0, 1},
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 1}
    };

    public static final int[] or_not3_outputs = new int[] {
            0, 1, 1, 1, 0, 0, 0, 0
    };
}
