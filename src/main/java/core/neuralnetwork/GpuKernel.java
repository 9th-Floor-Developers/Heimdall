package core.neuralnetwork;

import com.aparapi.Kernel;

public class GpuKernel extends Kernel {
	public float[] act, err, target, w, bias, wChange, bChange, gradChange, sq;
	
	public int op, n, batch, total, cur, prev, actCur, actPrev, wCur, bCur;
	public float lr;
	
	public GpuKernel(int actSize, int targetSize, int wSize, int bSize) {
		act = new float[actSize];
		err = new float[actSize];
		target = new float[targetSize];
		w = new float[wSize];
		wChange = new float[wSize];
		bias = new float[bSize];
		bChange = new float[bSize];
		gradChange = new float[bSize];
		sq = new float[bSize];
	}
	
	@Override
	public void run() {
		int g = getGlobalId();
		
		if (g < n) {
			if (op == 0) {
				calcValue(g);
			} else if (op == 1) {
				calcError(g);
			} else if (op == 2) {
				calcLayerErrors(g);
			} else if (op == 3) {
				calcWeightChange(g);
			} else if (op == 4) {
				calcWeightChangeGradient(g);
			} else if (op == 5) {
				applyWeightChangeGradient(g);
			} else if (op == 6) {
				applyWeightChange(g);
			} else if (op == 7) {
				focusOutputs(g);
			}
		}
	}
	
	private void applyWeightChange(int g) {
		int j = g / prev;
		
		w[wCur + g] += wChange[wCur + g] * (lr / sqrt(sq[bCur + j] + 1.0e-8f));
		wChange[wCur + g] = 0f;
	}
	
	private void applyWeightChangeGradient(int g) {
		float gs = gradChange[bCur + g];
		
		sq[bCur + g] = 0.9f * sq[bCur + g] + 0.1f * gs * gs;
		bias[bCur + g] += bChange[bCur + g] * lr;
		bChange[bCur + g] = 0f;
		gradChange[bCur + g] = 0f;
	}
	
	private void calcWeightChangeGradient(int g) {
		float gs = 0, s = 0;
		
		for (int i = 0; i < batch; i++) {
			int idx = i * total + actCur + g;
			float y = act[idx],
					e = err[idx];
			
			gs += e * y * (1f - y);
			s += e;
		}
		
		gradChange[bCur + g] += gs;
		bChange[bCur + g] += s;
	}
	
	private void calcWeightChange(int g) {
		int j = g / prev;
		int kk = g - j * prev;
		float s = 0;
		
		for (int i = 0; i < batch; i++) {
			int base = i * total;
			float y = act[base + actCur + j];
			s += err[base + actCur + j] * y * (1f - y) * act[base + actPrev + kk];
		}
		
		wChange[wCur + g] += s;
	}
	
	private void calcLayerErrors(int g) {
		int b = g / prev,
				kk = g - b * prev,
				base = b * total;
		float s = 0;
		
		for (int i = 0; i < cur; i++)
			s += err[base + actCur + i] * w[wCur + i * prev + kk];
		
		err[base + actPrev + kk] = s;
	}
	
	private void calcError(int g) {
		int b = g / cur,
				j = g - b * cur,
				idx = b * total + actCur + j;
		
		err[idx] = target[g] - act[idx];
	}
	
	private void calcValue(int g) {
		int b = g / cur,
				j = g - b * cur,
				base = b * total;
		float s = bias[bCur + j];
		
		for (int i = 0; i < prev; i++)
			s += act[base + actPrev + i] * w[wCur + j * prev + i];
		
		act[base + actCur + j] = 1f / (1f + exp(-s));
	}
	
	private void focusOutputs(int g) {
		int base = g * total + actCur, kk = 0;
		for (int j = 1; j < cur; j++)
			if (act[base + j] > act[base + kk])
				kk = j;
		for (int j = 0; j < cur; j++) {
			if (j == kk)
				act[base + j] = 1f;
			else
				act[base + j] = 0f;
		}
	}
}
