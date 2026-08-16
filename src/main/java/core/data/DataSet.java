package core.data;


import java.util.List;

public record DataSet(List<DataPoint> trainingDataPoints, List<DataPoint> testingDataPoints, int randomTrainingSize) {
    public DataSet(List<DataPoint> trainingDataPoints, List<DataPoint> testingDataPoints){
        this(trainingDataPoints, testingDataPoints, -1);
    }

    public void printSize(){
        String message = "Data set being created with training size " + getTrainingSize() + " and testing size " + getTestingSize();
        if (randomTrainingSize != -1){
            message += "\n random training size " + randomTrainingSize;
        }
        System.out.println(message);
    }

    public int getTrainingSize(){
        return trainingDataPoints().size();
    }

    public int getTestingSize(){
        return testingDataPoints.size();
    }

    public int getInputLength(){
        return trainingDataPoints.getFirst().getInputs().length;
    }

    public int getOutputLength(){
        return trainingDataPoints.getFirst().getTargetValues().length;
    }

    public int[] getLayerLengths(int[] hiddenLayerLengths){
        int[] layerLengths = new int[hiddenLayerLengths.length + 2];
        layerLengths[0] = getInputLength();
        layerLengths[layerLengths.length - 1] = getOutputLength();

        System.arraycopy(hiddenLayerLengths, 0, layerLengths, 1, hiddenLayerLengths.length);
        return layerLengths;
    }
}
