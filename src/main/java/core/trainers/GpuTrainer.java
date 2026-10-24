package core.trainers;

import com.aparapi.Range;
import core.data.DataPoint;
import core.data.DataSet;
import core.neuralnetwork.GpuKernel;
import core.neuralnetwork.NeuralNetwork;
import core.neuralnetwork.Neuron;

import java.util.Arrays;
import java.util.List;

public class GpuTrainer {
	private static final int LOCAL = 64;
	
	private final NeuralNetwork network;
	private final int[] sizes, actOff, wOff, bOff;
	private final int last, maxBatch, total;
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
		for (int i = 0; i < sizes.length; i++) {
			actOff[i] = a;
			a += sizes[i];
			if (i > 0) {
				wOff[i] = wSum;
				wSum += sizes[i] * sizes[i - 1];
				bOff[i] = bSum;
				bSum += sizes[i];
			}
		}
		total = a;
		
		k = new GpuKernel(maxBatch * total, maxBatch * sizes[last], wSum, bSum);
		k.total = total;
		k.setExplicit(true);
		uploadWeights();
	}
	
	public void uploadWeights() {
		for (int i = 1; i <= last; i++) {
			for (int j = 0; j < sizes[i]; j++) {
				Neuron neuron = network.getNeuron(i, j);
				k.bias[bOff[i] + j] = neuron.getBias();
				System.arraycopy(neuron.getWeights(), 0, k.w,
						wOff[i] + j * sizes[i - 1], sizes[i - 1]);
			}
		}
		Arrays.fill(k.wChange, 0f);
		Arrays.fill(k.bChange, 0f);
		Arrays.fill(k.gradChange, 0f);
		Arrays.fill(k.sq, 0f);
		k.put(k.w).put(k.bias).put(k.wChange).put(k.bChange).put(k.gradChange)
				.put(k.sq).put(k.act).put(k.err).put(k.target);
	}
	
	
	public void downloadWeights() {
		k.get(k.w).get(k.bias);
		
		for (int i = 1; i <= last; i++) {
			for (int j = 0; j < sizes[i]; j++) {
				Neuron neuron = network.getNeuron(i, j);
				neuron.setBias(k.bias[bOff[i] + j]);
				int from = wOff[i] + j * sizes[i - 1];
				neuron.setWeights(Arrays.copyOfRange(k.w, from, from + sizes[i - 1]));
			}
		}
	}
	
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
			
			for (int i = 0; i < count; i++) {
				DataPoint dp = points.get(start + i);
				in[i] = dp.inputs();
				System.arraycopy(dp.getTargetValues(), 0, k.target, i * outSize, outSize);
			}
			
			k.put(k.target);
			loadInputs(in, count);
			forward(count);
			
			if (focusOutput) {
				setLayer(last);
				launch(7, count);
			}
			
			k.get(k.act);
			for (int i = 0; i < count; i++) {
				int best = 0;
				for (int j = 1; j < outSize; j++)
					if (k.act[i * total + actOff[last] + j] > k.act[i * total + actOff[last] + best])
						best = j;
				
				if (best == points.get(start + i).targetResult())
					correct++;
			}
			
			for (int i = last; i >= 1; i--) {
				setLayer(i);
				if (i == last)
					launch(1, count * outSize);
				
				launch(3, sizes[i] * sizes[i - 1]);
				launch(4, sizes[i]);
				
				if (i > 1)
					launch(2, count * sizes[i - 1]);
			}
			
			if (needError) {
				k.get(k.err);
				for (int i = 0; i < count; i++) {
					float s = 0;
					for (int j = 0; j < outSize; j++)
						s += Math.abs(k.err[i * total + actOff[last] + j]);
					errorSum += s / outSize;
				}
			}
		}
		return new Stats(correct, errorSum);
	}
	
	/**
	 * Same as NeuralNetwork.applyWeightsChange(learningRate).
	 */
	public void applyWeights(float learningRate) {
		k.lr = learningRate;
		for (int i = 1; i <= last; i++) {
			setLayer(i);
			launch(5, sizes[i]);
			launch(6, sizes[i] * sizes[i - 1]);
		}
	}
	
	/**
	 * GPU version of NeuralNetwork.testAgent.
	 */
	public float test(DataSet dataSet) {
		float[][] in = new float[maxBatch][];
		int[] labels = new int[maxBatch];
		int count = 0, correct = 0;
		
		for (DataPoint dp : dataSet.testingDataPoints()) {
			in[count] = dp.inputs();
			labels[count] = dp.targetResult();
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
		for (int i = 0; i < count; i++) {
			int best = 0;
			for (int j = 1; j < sizes[last]; j++)
				if (k.act[i * total + actOff[last] + j] > k.act[i * total + actOff[last] + best])
					best = j;
			
			if (best == labels[i])
				correct++;
		}
		return correct;
	}
	
	private void loadInputs(float[][] inputs, int count) {
		for (int i = 0; i < count; i++)
			System.arraycopy(inputs[i], 0, k.act, i * total, sizes[0]);
		k.put(k.act);
	}
	
	private void forward(int count) {
		k.batch = count;
		for (int i = 1; i <= last; i++) {
			setLayer(i);
			launch(0, count * sizes[i]);
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
		int global = ((n + LOCAL - 1) / LOCAL) * LOCAL;
		k.execute(Range.create(global, LOCAL));
	}
	// endregion
}
