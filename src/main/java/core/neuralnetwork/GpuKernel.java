package core.neuralnetwork;

import com.aparapi.Kernel;
import core.trainers.GpuTrainer;

/**
 * The GPU math for the network, written as a single Aparapi {@link Kernel}.
 * <p>
 * A GPU can't work with objects, so this class
 * stores the whole network in flat {@code float[]} arrays instead.
 * <p>
 * Created and driven only by {@link GpuTrainer}.
 */
public class GpuKernel extends Kernel {
	public float[] act, err, target, w, bias, wChange, bChange, gradChange, sq;
	public int op, n, batch, total, cur, prev, actCur, actPrev, wCur, bCur;
	public float lr;
	
	/**
	 * Allocates all GPU buffers. Sizes come from {@link GpuTrainer}'s layout calculation.
	 *
	 * @param actSize    {@code maxBatch * total}, size of {@code act} and {@code err}
	 * @param targetSize {@code maxBatch * outputLayerSize}, size of {@code target}
	 * @param wSize      total number of weights in the network
	 * @param bSize      total number of non-input neurons, size of the bias and per-neuron arrays
	 */
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
	
	/**
	 * Runs on the GPU, once per thread. Thread id {@code g} is the element being handled.
	 * Which formula runs depends on {@link #op}:
	 * <ol start="0">
	 *     <li><b>Forward</b>, one thread per (sample, neuron): {@code sigmoid(bias + sum(prev * weight))}.
	 *     CPU equivalent: {@code Neuron.calcValue(Layer)}.</li>
	 *     <li><b>Output error</b>, one thread per (sample, output neuron): {@code target - value}.
	 *     CPU equivalent: {@code Neuron.calcError(float)}.</li>
	 *     <li><b>Error propagation</b>, one thread per (sample, previous-layer neuron): sums
	 *     {@code error * weight} over the current layer. CPU equivalent: {@code Layer.calcErrors(float, float[])}.</li>
	 *     <li><b>Weight changes</b>, one thread per weight: sums {@code error * sigmoidDerivative * prevValue}
	 *     over the batch into {@link #wChange}. CPU equivalent: the weight part of {@code Neuron.calcWeightChange(Layer)}.</li>
	 *     <li><b>Bias and gradient changes</b>, one thread per neuron: sums the gradient into {@link #gradChange}
	 *     and the raw error into {@link #bChange}. CPU equivalent: the gradient and bias part of
	 *     {@code Neuron.calcWeightChange(Layer)}.</li>
	 *     <li><b>Apply bias</b>, one thread per neuron: updates {@link #sq}, adds {@code bChange * lr} to the bias,
	 *     then clears the accumulators. CPU equivalent: the bias part of {@code Neuron.applyWeightChange(float)}.</li>
	 *     <li><b>Apply weights</b>, one thread per weight: adds {@code wChange * lr / sqrt(sq + 1e-8)} to the weight,
	 *     then clears {@code wChange}. CPU equivalent: the weight part of {@code Neuron.applyWeightChange(float)}.</li>
	 *     <li><b>Focus outputs</b>, one thread per sample: sets the largest output to 1 and the rest to 0.
	 *     CPU equivalent: {@code NeuralNetwork.focusOutputs(Neuron[])}.</li>
	 * </ol>
	 * Operation 6 must run after operation 5 of the same layer, because it reads the {@code sq} that operation 5 updates.
	 * <p>
	 * Aparapi translates this class's bytecode into OpenCL, so this method may only use primitives,
	 * primitive arrays, {@code if/else} and loops. No {@code switch}, lambdas, objects or strings.
	 */
	@Override
	public void run() {
		int g = getGlobalId();
		
		if (g < n) {
			// GPU cannot recognize switch statement, keep ifs
			if (op == 0) calcValue(g);
			else if (op == 1) calcError(g);
			else if (op == 2) calcLayerErrors(g);
			else if (op == 3) calcWeightChange(g);
			else if (op == 4) calcWeightChangeGradient(g);
			else if (op == 5) applyWeightChangeGradient(g);
			else if (op == 6) applyWeightChange(g);
			else if (op == 7) focusOutputs(g);
		}
	}
	
	// region Operations
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
		int j = g / prev, kk = g - j * prev;
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
		
		for (int j = 0; j < cur; j++)
			act[base + j] = j == kk ? 1f : 0f;
	}
	// endregion
}
