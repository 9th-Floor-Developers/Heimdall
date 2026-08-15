package core.data;

public class BasicDataPoint implements DataPoint {
    private final float[] inputs;
    private final int targetResult;

    public BasicDataPoint(float[] inputs, int targetResult) {
        this.inputs = inputs;
        this.targetResult = targetResult;
    }


    @Override
    public float[] getInputs() {
        return inputs;
    }

    @Override
    public int getTargetResult() {
        return targetResult;
    }
}
