package core.data;

import java.util.ArrayList;
import java.util.stream.Collectors;

abstract public class AbstractDataSetLoader {
    //TODO ADD DOCSTRINGS
    private int trainingSize = -1;
    private int testingSize = -1;
    protected int loadLimit;
    protected String src = null;

    public AbstractDataSetLoader setTrainingSize(int trainingSize) {
        if (trainingSize <= 0){
            throw new IllegalArgumentException("Training size must be greater than 0");
        }
        this.trainingSize = trainingSize;
        return this;
    }

    public AbstractDataSetLoader setTestingSize(int testingSize) {
        if (testingSize <= 0){
            throw new IllegalArgumentException("Testing size must be greater than 0");
        }
        this.testingSize = testingSize;
        return this;
    }

    public AbstractDataSetLoader setSrc(String src) {
        this.src = src;
        return this;
    }

    public AbstractDataSetLoader setTrainingSizeAsRemaining() {
        return setTrainingSize(Integer.MAX_VALUE);
    }

    public AbstractDataSetLoader setTestingSizeAsRemaining() {
        return setTestingSize(Integer.MAX_VALUE);
    }

    abstract protected ArrayList<DataPoint> loadDataPoints(int loadLimit) throws Exception;

    private Boolean usingRemainingSize(){
        return (trainingSize == Integer.MAX_VALUE || testingSize == Integer.MAX_VALUE);
    }

    public DataSet load() throws Exception {
        if (trainingSize == -1 || testingSize == -1){
            throw new IllegalStateException("Training size and/or testing size has been specified");
        }
        if (trainingSize == Integer.MAX_VALUE && testingSize == Integer.MAX_VALUE){
            throw new IllegalStateException("Remaining size is used as training size and testing size");
        }

        loadLimit = usingRemainingSize() ? Integer.MAX_VALUE : trainingSize + testingSize;
        ArrayList<DataPoint> dataPoints = loadDataPoints(loadLimit);

        if (dataPoints.size() < loadLimit && !usingRemainingSize()){
            throw new IllegalStateException("There is no few datapoints, compared to the combined data requested");
        }

        if (trainingSize == Integer.MAX_VALUE){
            trainingSize = dataPoints.size() - testingSize;
        }
        if (testingSize == Integer.MAX_VALUE){
            testingSize = dataPoints.size() - trainingSize;
        }

        if (trainingSize <= 0 || testingSize <= 0){
            throw new IllegalArgumentException("Training size and testing size must be greater than 0, " +
                    ", could be caused by the usage of size as remaining when there is no data left");
        }

        DataSet dataSet = new DataSet(
                new ArrayList<>(dataPoints.subList(0, trainingSize)),
                new ArrayList<>(dataPoints.subList(trainingSize, trainingSize + testingSize)));

        System.out.println("Data set being created with training size " + dataSet.getTrainingSize() + " and testing size " + dataSet.getTestingSize());
        return dataSet;
    }
}
