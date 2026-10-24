package core.trainers;

import com.aparapi.Range;
import core.data.DataPoint;
import core.data.DataSet;
import core.neuralnetwork.GpuKernel;
import core.neuralnetwork.NeuralNetwork;
import core.neuralnetwork.Neuron;

import java.util.Arrays;
import java.util.List;

/**
 * Runs training and testing of a {@link NeuralNetwork} on the GPU through {@link GpuKernel}.
 * <p>
 * The {@code NeuralNetwork}, {@code Layer} and {@code Neuron} classes are not modified and are not used
 * during training. At construction the trainer copies their weights and biases into flat GPU buffers
 * ({@link #uploadWeights()}); training then happens only on the GPU, and {@link #downloadWeights()} must be
 * called to copy the result back before the network is saved or used on the CPU.
 * <p>
 * CPU to GPU equivalents:
 * <ul>
 *     <li>{@code NeuralNetwork.calcOutputs} + {@code backProp} -> {@link #accumulate}</li>
 *     <li>{@code NeuralNetwork.applyWeightsChange} -> {@link #applyWeights}</li>
 *     <li>{@code NeuralNetwork.testAgent} -> {@link #test}</li>
 * </ul>
 * Images are processed in chunks of at most {@code maxBatch}, which only limits GPU memory use. It does not
 * change the result, because gradients are summed and applied once per round, exactly like the CPU code.
 */
public class GpuTrainer {
	private static final int LOCAL = 64;
	
	private final NeuralNetwork network;
	private final int[] sizes, actOff, wOff, bOff;
	private final int last, maxBatch, total;
	private final GpuKernel k;
	
	/**
	 * Builds the buffer layout for {@code network}, creates the {@link GpuKernel}, and uploads the network's
	 * current weights and biases to the GPU.
	 * <p>
	 * Create it after the network exists, and after loading a saved agent if one is used.
	 *
	 * @param network  network to train; its weights are copied to the GPU now and back on {@link #downloadWeights()}
	 * @param maxBatch maximum images per GPU pass; larger is faster but uses more GPU memory
	 *                 ({@code maxBatch * total} floats per activation buffer)
	 */
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
	
	/**
	 * Copies weights and biases from the {@code Neuron} objects to the GPU, and resets the pending gradient
	 * accumulators and the squared-gradient state to 0.
	 * <p>
	 * Called automatically by the constructor. Call it again if the weights are changed on the CPU while
	 * the trainer exists, for example after loading an agent.
	 */
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
	
	/**
	 * Copies the trained weights and biases from the GPU back into the {@code Neuron} objects
	 * ({@code Neuron.setWeights} and {@code Neuron.setBias}).
	 * <p>
	 * Must be called before saving the agent ({@code DataLogger.saveAgent}) or using any CPU method of the
	 * network, otherwise those still hold the untrained values. The squared-gradient state is not copied
	 * back, since {@code Neuron} has no setter for it.
	 */
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
	 * Runs the forward pass and backpropagation for all given points and adds their gradients to the GPU
	 * accumulators. Weights are not changed until {@link #applyWeights(float)} is called.
	 * <p>
	 * CPU equivalent: the loop in {@code FeedForwardTrainer.trainAgentRound} that calls
	 * {@code agent.calcOutputs(...)}, counts correct guesses, and calls {@code agent.backProp(...)}.
	 * Unlike the CPU loop, backprop always runs here; {@code needError} only controls whether the error is
	 * read back.
	 *
	 * @param points      training points; processed in chunks of at most {@code maxBatch}
	 * @param focusOutput if true, the outputs are set to 1 for the largest and 0 for the rest before backprop
	 *                    (same as {@code calcOutputs(inputs, true)})
	 * @param needError   if true, the output errors are copied back to compute {@link Stats#errorSum()},
	 *                    which is slightly slower
	 * @return how many points were guessed correctly and the summed error
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
	 * Applies the gradients collected by {@link #accumulate} to all weights and biases, then clears them.
	 * <p>
	 * CPU equivalent: {@code NeuralNetwork.applyWeightsChange(float)}, which runs
	 * {@code Neuron.applyWeightChange(float)} on every neuron.
	 *
	 * @param learningRate step size; {@code FeedForwardTrainer} passes {@code learningRate / trainingSize}
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
	 * Measures accuracy on the testing points, using the GPU for the forward passes.
	 * <p>
	 * CPU equivalent: {@code NeuralNetwork.testAgent(DataSet, boolean)}. {@code focusOutputs} is not needed
	 * because the highest output is the same with or without it.
	 *
	 * @param dataSet data set whose {@code testingDataPoints()} are used
	 * @return fraction of testing points guessed correctly (0.0 to 1.0)
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
	
	/**
	 * Describes the device the kernel actually runs on, for checking that the GPU is used.
	 * <p>
	 * Only meaningful after the first launch; before that Aparapi has not chosen a device yet.
	 *
	 * @return the device name, e.g. the GPU's name, or a Java thread pool if Aparapi fell back to the CPU
	 */
	public String getExecutionMode() {
		return String.valueOf(k.getTargetDevice());
	}
	
	/**
	 * Releases the GPU buffers and native resources. Call once when training is finished;
	 * the trainer can't be used afterward.
	 */
	public void dispose() {
		k.dispose();
	}
	
	// region internals
	/**
	 * Forward-passes one chunk of testing images and counts the correct guesses.
	 *
	 * @param in     input arrays of the chunk
	 * @param labels expected digit for each image
	 * @param count  number of valid entries in {@code in} and {@code labels}
	 * @return number of images where the highest output neuron matches the label
	 */
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
	
	/**
	 * Writes the input pixels of a chunk into the input-layer part of the activation buffer and uploads it.
	 * <p>
	 * CPU equivalent: the {@code neuron.setValue(inputs[j])} step for layer 0 in {@code calcOutputs}.
	 *
	 * @param inputs input arrays, one per image
	 * @param count  number of images in the chunk
	 */
	private void loadInputs(float[][] inputs, int count) {
		for (int i = 0; i < count; i++)
			System.arraycopy(inputs[i], 0, k.act, i * total, sizes[0]);
		k.put(k.act);
	}
	
	/**
	 * Runs the forward pass for layers 1 to the output layer on the GPU, using the inputs from {@link #loadInputs}.
	 * <p>
	 * CPU equivalent: the layer loop in {@code NeuralNetwork.calcOutputs}.
	 *
	 * @param count number of images in the current chunk
	 */
	private void forward(int count) {
		k.batch = count;
		for (int i = 1; i <= last; i++) {
			setLayer(i);
			launch(0, count * sizes[i]);
		}
	}
	
	/**
	 * Points the kernel at layer {@code l}: sets its size, the previous layer's size, and the buffer offsets.
	 * <p>
	 * Must be called before every launch that works on a layer.
	 *
	 * @param l layer index (1 to output layer)
	 */
	private void setLayer(int l) {
		k.cur = sizes[l];
		k.prev = sizes[l - 1];
		k.actCur = actOff[l];
		k.actPrev = actOff[l - 1];
		k.wCur = wOff[l];
		k.bCur = bOff[l];
	}
	
	/**
	 * Starts one GPU launch of {@code op} with {@code n} useful threads.
	 * The thread count is rounded up to a multiple of the work-group size (64); the extra threads
	 * are ignored by the kernel through its {@code g < n} check.
	 *
	 * @param op operation number, see {@link GpuKernel#run()}
	 * @param n  how many elements the operation handles
	 */
	private void launch(int op, int n) {
		k.op = op;
		k.n = n;
		int global = ((n + LOCAL - 1) / LOCAL) * LOCAL;
		k.execute(Range.create(global, LOCAL));
	}
	// endregion
}
