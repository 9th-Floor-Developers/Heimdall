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

    public int[] getLayerLengths(int[] hiddenLayerLengths){
        int[] layerLengths = new int[hiddenLayerLengths.length + 2];
        layerLengths[0] = getInputLength();
        layerLengths[layerLengths.length - 1] = getOutputLength();

        System.arraycopy(hiddenLayerLengths, 0, layerLengths, 1, hiddenLayerLengths.length);
        return layerLengths;
    }
}
