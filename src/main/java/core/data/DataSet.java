package core.data;


import core.trainers.TrainingRoundResult;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;

public record DataSet(
		List<DataPoint> allTrainingDataPoints,
		List<DataPoint> testingDataPoints,
		@Nullable BiFunction<List<DataPoint>, TrainingRoundResult, List<DataPoint>> trainingFilter
) {
	public DataSet(List<DataPoint> trainingDataPoints, List<DataPoint> testingDataPoints) {
		this(trainingDataPoints, testingDataPoints, null);
	}
	
	public void printSize() {
		String message = "Data set being created with training size " + getTrainingDataPoints(null).size() + " and testing size " + getTestingSize();
		if (trainingFilter != null)
			message += "\n from all training size " + allTrainingDataPoints.size();
		System.out.println(message);
	}
	
	public List<DataPoint> getTrainingDataPoints(@Nullable TrainingRoundResult trainingRoundResult) {
		if (trainingFilter == null)
			return allTrainingDataPoints;
		else {
			ArrayList<DataPoint> newList = new ArrayList<>(allTrainingDataPoints);
			Collections.shuffle(newList);
			
			return newList.subList(0, trainingFilter.apply(newList, null).size());
		}
	}
	
	public int getTestingSize() {
		return testingDataPoints.size();
	}
	
	public int getInputLength() {
		return allTrainingDataPoints.getFirst().inputs().length;
	}
	
	public int getOutputLength() {
		return allTrainingDataPoints.getFirst().getTargetValues().length;
	}
	
	public int[] getLayerLengths(int[] hiddenLayerLengths) {
		int[] layerLengths = new int[hiddenLayerLengths.length + 2];
		layerLengths[0] = getInputLength();
		layerLengths[layerLengths.length - 1] = getOutputLength();
		
		System.arraycopy(hiddenLayerLengths, 0, layerLengths, 1, hiddenLayerLengths.length);
		return layerLengths;
	}
}
