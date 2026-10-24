package core.trainers;

import com.aparapi.Range;
import core.data.DataPoint;
import core.data.DataSet;
import core.neuralnetwork.GpuKernel;
import core.neuralnetwork.NeuralNetwork;
import core.neuralnetwork.Neuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GpuTrainer {
	private static final int LOCAL = 64;
	
	private final NeuralNetwork network;
	private final int[] sizes;
	private final int last, maxBatch, total;
	private final int[] actOff, wOff, bOff;
	private final GpuKernel k;
	
	public GpuTrainer(NeuralNetwork network, int maxBatch) {
		this.network = network;
		this.maxBatch = maxBatch;
		this.sizes = network.getLayerLengths();
		this.last = sizes.length - 1;
		
		actOff = new int[sizes.length];
		wOff = new int[sizes.length];
		bOff = new int[sizes.length];
		int a = 0, wSum = 0, bSum = 0;
		for (int l = 0; l < sizes.length; l++) {
			actOff[l] = a;
			a += sizes[l];
			if (l > 0) {
				wOff[l] = wSum;
				wSum += sizes[l] * sizes[l - 1];
				bOff[l] = bSum;
				bSum += sizes[l];
			}
		}
		total = a;
		
		k = new GpuKernel(maxBatch * total, maxBatch * sizes[last], wSum, bSum);
		k.total = total;
		k.setExplicit(true);   // we control host<->GPU copies
		uploadWeights();
	}
	
	/** Network objects -> GPU. Call again if you load a saved agent or change weights on the CPU. */
	public void uploadWeights() {
		for (int l = 1; l <= last; l++) {
			for (int j = 0; j < sizes[l]; j++) {
				Neuron neuron = network.getNeuron(l, j);
				k.bias[bOff[l] + j] = neuron.getBias();
				System.arraycopy(neuron.getWeights(), 0, k.w, wOff[l] + j * sizes[l - 1], sizes[l - 1]);
			}
		}
		Arrays.fill(k.wChange, 0f);
		Arrays.fill(k.bChange, 0f);
		Arrays.fill(k.gradChange, 0f);
		Arrays.fill(k.sq, 0f);
		k.put(k.w).put(k.bias).put(k.wChange).put(k.bChange).put(k.gradChange)
				.put(k.sq).put(k.act).put(k.err).put(k.target);
	}
	
	/** GPU -> network objects. MUST be called before saving the agent or using CPU methods. */
	public void downloadWeights() {
		k.get(k.w).get(k.bias);
		for (int l = 1; l <= last; l++) {
			for (int j = 0; j < sizes[l]; j++) {
				Neuron neuron = network.getNeuron(l, j);
				neuron.setBias(k.bias[bOff[l] + j]);
				int from = wOff[l] + j * sizes[l - 1];
				neuron.setWeights(Arrays.copyOfRange(k.w, from, from + sizes[l - 1]));
			}
		}
	}
	
	/**
	 * Forward + backprop + weight update for one batch (inputs.length <= maxBatch).
	 * With a batch of 1 this matches calcOutputs + backProp + applyWeightsChange.
	 *
	 * @return mean absolute output error of the batch
	 */
//	public float trainBatch(float[][] inputs, float[][] targets, float learningRate) {
//		int count = inputs.length;
//		int outSize = sizes[last];
//
//		loadInputs(inputs, count);
//		for (int b = 0; b < count; b++)
//			System.arraycopy(targets[b], 0, k.target, b * outSize, outSize);
//		k.put(k.target);
//
//		forward(count);
//
//		for (int l = last; l >= 1; l--) {
//			setLayer(l);
//			if (l == last) launch(1, count * outSize);
//			launch(3, sizes[l] * sizes[l - 1]);
//			launch(4, sizes[l]);
//			if (l > 1) launch(2, count * sizes[l - 1]);
//		}
//
//		k.get(k.err);
//		float sum = 0;
//		for (int b = 0; b < count; b++)
//			for (int j = 0; j < outSize; j++)
//				sum += Math.abs(k.err[b * total + actOff[last] + j]);
//
//		k.lr = learningRate;
//		for (int l = 1; l <= last; l++) {
//			setLayer(l);
//			launch(5, sizes[l]);
//			launch(6, sizes[l] * sizes[l - 1]);
//		}
//		return sum / (count * outSize);
//	}
	
	public record Stats(int correct, float errorSum) {}
	
	/**
	 * Forward + backprop over all points, in chunks. Gradients pile up on the GPU,
	 * nothing is applied until {@link #applyWeights(float)}.
	 */
	public Stats accumulate(List<DataPoint> points, boolean focusOutput, boolean needError) {
		int outSize = sizes[last];
		float[][] in = new float[maxBatch][];
		int correct = 0;
		float errorSum = 0;
		
		for (int start = 0; start < points.size(); start += maxBatch) {
			int count = Math.min(maxBatch, points.size() - start);
			
			for (int b = 0; b < count; b++) {
				DataPoint dp = points.get(start + b);
				in[b] = dp.inputs();
				System.arraycopy(dp.getTargetValues(), 0, k.target, b * outSize, outSize);
			}
			k.put(k.target);
			loadInputs(in, count);
			forward(count);
			
			if (focusOutput) {
				setLayer(last);
				launch(7, count);
			}
			
			// training score (same as the maxIndex loop in your trainer)
			k.get(k.act);
			for (int b = 0; b < count; b++) {
				int best = 0;
				for (int j = 1; j < outSize; j++)
					if (k.act[b * total + actOff[last] + j] > k.act[b * total + actOff[last] + best])
						best = j;
				if (best == (int) points.get(start + b).targetResult())
					correct++;
			}
			
			// backprop, gradients accumulate on the GPU
			for (int l = last; l >= 1; l--) {
				setLayer(l);
				if (l == last) launch(1, count * outSize);
				launch(3, sizes[l] * sizes[l - 1]);
				launch(4, sizes[l]);
				if (l > 1) launch(2, count * sizes[l - 1]);
			}
			
			if (needError) {                // sum of per-sample average |error|
				k.get(k.err);
				for (int b = 0; b < count; b++) {
					float s = 0;
					for (int j = 0; j < outSize; j++)
						s += Math.abs(k.err[b * total + actOff[last] + j]);
					errorSum += s / outSize;
				}
			}
		}
		return new Stats(correct, errorSum);
	}
	
	/** Same as NeuralNetwork.applyWeightsChange(learningRate). */
	public void applyWeights(float learningRate) {
		k.lr = learningRate;
		for (int l = 1; l <= last; l++) {
			setLayer(l);
			launch(5, sizes[l]);
			launch(6, sizes[l] * sizes[l - 1]);
		}
	}
	
	/** GPU version of NeuralNetwork.testAgent (no focusOutputs needed, argmax is the same). */
	public float test(DataSet dataSet) {
		float[][] in = new float[maxBatch][];
		int[] labels = new int[maxBatch];
		int count = 0, correct = 0;
		
		for (DataPoint dp : dataSet.testingDataPoints()) {
			in[count] = dp.inputs();
			labels[count] = (int) dp.targetResult();
			count++;
			if (count == maxBatch) {
				correct += evaluate(in, labels, count);
				count = 0;
			}
		}
		if (count > 0)
			correct += evaluate(in, labels, count);
		
		return (float) correct / dataSet.getTestingSize();
	}
	
	public String getExecutionMode() {
		return String.valueOf(k.getTargetDevice());
	}
	
	public void dispose() {
		k.dispose();
	}
	
	// region internals
	private int evaluate(float[][] in, int[] labels, int count) {
		loadInputs(in, count);
		forward(count);
		k.get(k.act);
		
		int correct = 0;
		for (int b = 0; b < count; b++) {
			int best = 0;
			for (int j = 1; j < sizes[last]; j++)
				if (k.act[b * total + actOff[last] + j] > k.act[b * total + actOff[last] + best])
					best = j;
			if (best == labels[b]) correct++;
		}
		return correct;
	}
	
	private void loadInputs(float[][] inputs, int count) {
		for (int b = 0; b < count; b++)
			System.arraycopy(inputs[b], 0, k.act, b * total, sizes[0]);
		k.put(k.act);
	}
	
	private void forward(int count) {
		k.batch = count;
		for (int l = 1; l <= last; l++) {
			setLayer(l);
			launch(0, count * sizes[l]);
		}
	}
	
	private void setLayer(int l) {
		k.cur = sizes[l];
		k.prev = sizes[l - 1];
		k.actCur = actOff[l];
		k.actPrev = actOff[l - 1];
		k.wCur = wOff[l];
		k.bCur = bOff[l];
	}
	
	private void launch(int op, int n) {
		k.op = op;
		k.n = n;
		int global = ((n + LOCAL - 1) / LOCAL) * LOCAL;   // round up, kernel ignores extra threads
		k.execute(Range.create(global, LOCAL));
	}
	// endregion
}
