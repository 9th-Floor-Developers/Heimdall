package core.data;


import java.util.List;

public record DataSet(List<DataPoint> trainingDataPoints, List<DataPoint> testingDataPoints) {
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
