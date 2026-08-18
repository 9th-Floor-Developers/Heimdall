package core.utils;

import java.text.DecimalFormat;
import java.util.Arrays;

public class DataUtils {
	public static int universalSeed = 116117102;
	
	public static float getAverage(float[] array) {
		float sum = 0;
		for (float element : array)
			sum += element;
		return sum / array.length;
	}
	
	public static float getAverage(Float[] array) {
		float sum = 0;
		for (float element : array)
			sum += element;
		return sum / array.length;
	}
	
	
	public static String formatToPercent(int x, int max) {
		return formatToPercent(x, max);
	}
	
	public static String formatToPercent(float x, float max) {
		return formatToPercent(x / max * 100);
	}
	
	public static String formatToPercent(float factor) {
		return new DecimalFormat("###.##").format(factor * 100) + "%";
	}
	
	public static String arrayToStats(Float[] array, boolean isFactor) {
		Arrays.sort(array);
		float average = getAverage(array), high = array[array.length - 1], low = array[0];
		
		if (isFactor)
			return "|H: " + formatToPercent(high) + "| |L: " + formatToPercent(low)
					+ "| |AVG: " + formatToPercent(average) + "|";
		else
			return "|H: " + high + "| |L: " + low + "| |AVG: " + average + "|";
	}
}
