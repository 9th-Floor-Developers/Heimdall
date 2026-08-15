package core.data;

public interface DataPoint{
    float[] getInputs();
    default float[] getTargetValues(){
        float[] target = new float[10];  // 10 representing digits 0-9
        target[getTargetResult()] = 1;
        return target;
    }
    int getTargetResult();
}
