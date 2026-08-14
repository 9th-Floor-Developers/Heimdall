package utils;

public class DataUtils {
    public static int universalSeed = 116117102;

    public static float getAverage(float[] array){
        float sum = 0;

        for (float element : array) {
            sum += element;
        }
        return sum / array.length;
    }
}
