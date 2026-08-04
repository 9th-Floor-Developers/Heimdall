package utils;

public class DataUtils {
    public static float getAverage(float[] array){
        float sum = 0;

        for (float element : array) {
            sum += element;
        }
        return sum / array.length;
    }
}
