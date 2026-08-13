package data.base;

import java.util.List;

public record DataSet(List<DataPoint> dataPoints) {
    public int getSize(){
        return dataPoints().size();
    }

    public int getInputLength(){
        return dataPoints.getFirst().getInputs().length;
    }

    public int getOutputLength(){
        return dataPoints.getFirst().getTargetValues().length;
    }
}
