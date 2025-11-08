import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class Trainer {
	private final NeuralNetwork[] agents;  // all agents to be used in training session
	
	/**
	 * Trainer and all agents within.
	 *
	 * @param agentsPerRound number of agents to train with per round
	 * @param layerLengths   arraylist containing layer lengths (layerLengths.length will be used to create number of layers)
	 */
	public Trainer(int agentsPerRound, int[] layerLengths) {
		System.out.println("Initializing Agents...");
		
		agents = new NeuralNetwork[agentsPerRound];
		for (int i = 0; i < agentsPerRound; i++)
			// new network with layer structure as specified in layers variable
			agents[i] = new NeuralNetwork(layerLengths);
	}
	
	/**
	 * Trains all agents using deep learning, based on input dataset and desired outputs, repeats training for number of rounds
	 *
	 * @param inputs 2D array of floats or "flat inputs" representing a piece of data
	 * @param outputs 2D array of float outputs representing all the desired outputs per image
	 * @param numRounds number of rounds per training session
	 * @param scale range to randomly modify weights, should be between 0.0-0.5
	 * @throws InterruptedException if thread error occurs
	 */
	public void train(float[][] inputs, int[] outputs, int numRounds, float scale) throws InterruptedException {
		System.out.println("Training...");
		
		AtomicInteger bestScore = new AtomicInteger(-1);
		AtomicReference<NeuralNetwork> bestAgent = new AtomicReference<>(agents[0]);
		ArrayList<Thread> threads = new ArrayList<>();
		for (int i = 1; i <= numRounds; i++) {
			AtomicInteger roundBest = new AtomicInteger(-1);
			for (NeuralNetwork agent : agents) {
				Thread thread = new Thread(() -> {
					int score = eval(agent, inputs, outputs);
					
					if (score > bestScore.get()) {
						bestScore.set(score);
						bestAgent.set(agent);
					}
					if (score > roundBest.get())
						roundBest.set(score);
				});
				thread.start();
				threads.add(thread);
			}
			
			for (Thread thread : threads)
				thread.join();
			threads.clear();
			
			float percent = ((float) roundBest.get() / outputs.length) * 100;
			String formatted = new DecimalFormat("#.##").format(percent);
			
			System.out.println(i + ": [" + roundBest.get() + "/" + inputs.length + "] (" + formatted + "%)");
			
			for (int j = 0; j < agents.length; j++)
				agents[j] = bestAgent.get().evolve(scale);
		}
		
		System.out.println("Best: " + bestScore.get() + "%");
	}
	
	
	/**
	 * Evaluates the accuracy of a single network.
	 * <p>
	 * This should be run in a thread for fastest results.
	 *
	 * @param network network to evaluate
	 * @param inputs 2D array of floats or "flat inputs" representing a piece of data
	 * @param outputs 2D array of float outputs representing all the desired outputs per image
	 * @return how many data sets were guessed correctly
	 */
	private int eval(NeuralNetwork network, float[][] inputs, int[] outputs) {
		int score = 0;
		for (int i = 0; i < inputs.length; i++) {
			ArrayList<Float> calculatedOutputs = network.calculate(inputs[i]);
			
			Optional<Float> max = calculatedOutputs.stream().max(Float::compareTo);
			if (max.isEmpty())
				continue;
			
			if (outputs[i] == calculatedOutputs.indexOf(max.get()))
				score++;
		}
		
		return score;
	}
}
