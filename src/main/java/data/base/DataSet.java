package data.base;

import java.util.ArrayList;

public record DataSet(ArrayList<DataPoint> trainingDataPoints, ArrayList<DataPoint> testingDataPoints) {
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
