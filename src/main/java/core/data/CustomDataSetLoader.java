package core.data;

import java.util.List;

public class CustomDataSetLoader extends AbstractDataSetLoader{
    private final List<DataPoint> dataPoints;


    public CustomDataSetLoader(List<DataPoint> dataPoints) {
        this.dataPoints = dataPoints;
    }

    public static CustomDataSetLoader loadFromList(List<DataPoint> dataPoints){
        return new CustomDataSetLoader(dataPoints);
    }

    @Override
    protected List<DataPoint> loadDataPoints(int loadLimit) throws Exception {
        return dataPoints;
    }
}
