package legacy.matrixices;

public class MatrixUtils {
	
	static float[][]  multiplyMatrix(float[][] a, float[][] b)
	{
		int i, j, k;
		
		float[][] c = new float[a.length][b[0].length];
		
		for (i = 0; i < a.length; i++) {
			for (j = 0; j < b[0].length; j++) {
				for (k = 0; k < b.length; k++)
					c[i][j] += a[i][k] * b[k][j];
			}
		}
		
		return c;
	}
	
}
