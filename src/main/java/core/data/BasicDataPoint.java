package core.data;

public record BasicDataPoint(float[] inputs,
                             int targetResult) implements DataPoint {}
