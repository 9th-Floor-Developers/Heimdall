package core.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

abstract public class AbstractDataSetLoader {
    //TODO ADD DOCSTRINGS
    private int allTrainingSize = -1;
    private int testingSize = -1;
    private int randomTrainingSize =  -1;
    protected int loadLimit;
    protected String src = null;

    public AbstractDataSetLoader setAllTrainingSize(int allTrainingSize) {
        if (allTrainingSize <= 0){
            throw new IllegalArgumentException("Training size must be greater than 0");
        }
        this.allTrainingSize = allTrainingSize;
        return this;
    }

    public AbstractDataSetLoader setRandomTrainingSize(int randomTrainingSize) {
        if (randomTrainingSize <= 0){
            throw new IllegalArgumentException("Random training size must be greater than 0");
        }
        if (randomTrainingSize > allTrainingSize){
            throw new IllegalArgumentException("Training size must be set before random training Size and Random training size must be less than training size");
        }
        this.randomTrainingSize = randomTrainingSize;
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
        return setAllTrainingSize(Integer.MAX_VALUE);
    }

    public AbstractDataSetLoader setTestingSizeAsRemaining() {
        return setTestingSize(Integer.MAX_VALUE);
    }

    abstract protected List<DataPoint> loadDataPoints(int loadLimit) throws Exception;

    private Boolean usingRemainingSize(){
        return (allTrainingSize == Integer.MAX_VALUE || testingSize == Integer.MAX_VALUE);
    }

    public DataSet load() throws Exception {
        if (allTrainingSize == -1 || testingSize == -1){
            throw new IllegalStateException("Training size and/or testing size has been specified");
        }
        if (allTrainingSize == Integer.MAX_VALUE && testingSize == Integer.MAX_VALUE){
            throw new IllegalStateException("Remaining size is used as training size and testing size");
        }

        loadLimit = usingRemainingSize() ? Integer.MAX_VALUE : allTrainingSize + testingSize;
        ArrayList<DataPoint> dataPoints = new ArrayList<>(loadDataPoints(loadLimit));

        if (dataPoints.size() < loadLimit && !usingRemainingSize()){
            throw new IllegalStateException("There is no few datapoints, compared to the combined data requested");
        }

        if (allTrainingSize == Integer.MAX_VALUE){
            allTrainingSize = dataPoints.size() - testingSize;
        }
        if (testingSize == Integer.MAX_VALUE){
            testingSize = dataPoints.size() - allTrainingSize;
        }

        if (allTrainingSize <= 0 || testingSize <= 0){
            throw new IllegalArgumentException("Training size and testing size must be greater than 0, " +
                    ", could be caused by the usage of size as remaining when there is no data left");
        }

        Collections.shuffle(dataPoints);

        return new DataSet(
                dataPoints.subList(0, allTrainingSize),
                dataPoints.subList(allTrainingSize, allTrainingSize + testingSize),
                randomTrainingSize);
    }
}
