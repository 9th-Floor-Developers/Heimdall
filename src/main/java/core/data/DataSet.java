package core.data;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public record DataSet(List<DataPoint> allTrainingDataPoints, List<DataPoint> testingDataPoints, int randomTrainingSize) {
    public DataSet(List<DataPoint> trainingDataPoints, List<DataPoint> testingDataPoints){
        this(trainingDataPoints, testingDataPoints, -1);
    }

    public void printSize(){
        String message = "Data set being created with training size " + getTrainingSize() + " and testing size " + getTestingSize();
        if (randomTrainingSize != -1){
            message += "\n from all training size " + allTrainingDataPoints.size();
        }
        System.out.println(message);
    }

    public int getTrainingSize(){
        if (randomTrainingSize == -1){
            return allTrainingDataPoints.size();
        }
        else {
           return randomTrainingSize;
        }
    }

    public List<DataPoint> getTrainingDataPoints(){
        if (randomTrainingSize == -1){
            return allTrainingDataPoints;
        }
        else {
            ArrayList<DataPoint> newList = new ArrayList<>(allTrainingDataPoints);
            Collections.shuffle(newList);

            return newList.subList(0, randomTrainingSize);
        }
    }

    public int getTestingSize(){
        return testingDataPoints.size();
    }

    public int getInputLength(){
        return allTrainingDataPoints.getFirst().getInputs().length;
    }

    public int getOutputLength(){
        return allTrainingDataPoints.getFirst().getTargetValues().length;
    }


    public int[] getLayerLengths(int[] hiddenLayerLengths){
        int[] layerLengths = new int[hiddenLayerLengths.length + 2];
        layerLengths[0] = getInputLength();
        layerLengths[layerLengths.length - 1] = getOutputLength();

        System.arraycopy(hiddenLayerLengths, 0, layerLengths, 1, hiddenLayerLengths.length);
        return layerLengths;
    }
}
