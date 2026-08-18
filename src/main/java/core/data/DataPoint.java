package core.data;

public interface DataPoint {
	float[] inputs();
	
	default float[] getTargetValues() {
		float[] target = new float[10];  // 10 representing digits 0-9
		target[targetResult()] = 1;
		return target;
	}
	
	int targetResult();
}
