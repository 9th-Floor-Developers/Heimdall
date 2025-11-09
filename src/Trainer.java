import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReferenceArray;

public class Trainer {
	private NeuralNetwork[] agents;  // all agents to be used in training session
	
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
		
		
		AtomicInteger bestScore = new AtomicInteger(0);
		AtomicReferenceArray<NeuralNetwork> elders = new AtomicReferenceArray<>(5);
		AtomicIntegerArray elderScores = new AtomicIntegerArray(5);
		ArrayList<Thread> threads = new ArrayList<>();
		NeuralNetwork[] newGeneration = new NeuralNetwork[agents.length];
		
		System.out.println("elder scores: " + elderScores);
		
		for (int i = 0; i < elders.length(); i++)
			elders.set(i, agents[i]);
		
		for (int i = 1; i <= numRounds; i++) {
//			System.out.println(Arrays.toString(agents));
			AtomicInteger roundBest = new AtomicInteger(0);
			for (NeuralNetwork agent : agents) {
//				if (agent == null){
//					System.out.println("continuing");
//					continue;
//				}
				Thread thread = new Thread(() -> {
					// region elders
					int score = eval(agent, inputs, outputs);  // number of values guessed correctly (0-# Values)

					if (score > bestScore.get())
						bestScore.set(score);
					if (score > roundBest.get())
						roundBest.set(score);

					int length = elderScores.length();

					int insertIndex = -1;
					for (int j = 0; j < length; j++) {
						if (score > elderScores.get(j)) {
							insertIndex = j;
							break;
						}
					}
					
					// default value
					

					if (insertIndex == -1)
						return;

					for (int j = length - 1; j > insertIndex; j--) {
						elderScores.set(j, elderScores.get(j - 1));
//						System.out.println("A " + elders.get(j - 1));
						elders.set(j, elders.get(j - 1));
					}

					elderScores.set(insertIndex, score);
//					System.out.println("B " + agent);
					elders.set(insertIndex, agent);
					// endregion
					
					// region averaged crossbred children
					// endregion
					
					// region new blood
					// endregion
				});
				thread.start();
				threads.add(thread);
			}
			
			for (Thread thread : threads)
				thread.join();
			threads.clear();
			
			
			NeuralNetwork[] randomCrossbredChildren = new NeuralNetwork[5];
			// region random crossbred children
			for (int j = 0; j < randomCrossbredChildren.length; j++) {
				NeuralNetwork parentA = elders.get(randomIndex(elders.length())),
						parentB = elders.get(randomIndex(elders.length()));
				Layer[] parentALayers = parentA.getLayers(),
						parentBLayers = parentB.getLayers(),
						childLayers = new Layer[parentALayers.length];
				
				for (int k = 0; k < parentALayers.length; k++) {
					Layer parentALayer = parentALayers[k],
							parentBLayer = parentBLayers[k],
							childLayer = new Layer(k, parentA, parentALayer.getNumNeurons());
					
					for (int l = 0; l < parentALayer.getNumNeurons(); l++) {
						Neuron parentANeuron = parentALayer.getNeuron(l),
								parentBNeuron = parentBLayer.getNeuron(l),
								childNeuron = new Neuron(parentANeuron.getNumWeights());
						
						for (int n = 0; n < parentANeuron.getNumWeights(); n++) {
							float parentANeuronWeight = parentANeuron.getWeight(n),
									parentBNeuronWeight = parentBNeuron.getWeight(n),
									childWeight = (Math.random() < .5f) ? parentANeuronWeight : parentBNeuronWeight;
							childNeuron.setWeight(n, childWeight);
						}
						
						float value = (Math.random() < .5f) ? parentANeuron.getValue() : parentBNeuron.getValue();
						childNeuron.setValue(value);
						
						childLayer.setNeuron(l, childNeuron);
					}
					
					childLayers[k] = childLayer;
				}
				NeuralNetwork child = new NeuralNetwork(childLayers);
//				System.out.println("network " + j + ": " + child);
				randomCrossbredChildren[j] = child;
			}
			// endregion
			
//			System.out.println("random crossbred children: " + Arrays.toString(randomCrossbredChildren));
			
			float percent = ((float) roundBest.get() / outputs.length) * 100;
			String formatted = new DecimalFormat("#.##").format(percent);
			
			System.out.println(i + ": [" + roundBest.get() + "/" + inputs.length + "] (" + formatted + "%)");
			
			
//			System.out.println("elders: " + elders);
			for (int j = 0; j < 5; j++)
				newGeneration[j] = elders.get(j);
			
//			System.out.println("children: " + Arrays.toString(randomCrossbredChildren));
			
			for (int j = 5; j < 10; j++)
				newGeneration[j] = randomCrossbredChildren[j - 5];
			
//			System.out.println(Arrays.toString(newGeneration));
			
			// region mutant elders
			for (int j = 10; j < 15; j++)
				newGeneration[j] = elders.get(j - 10).evolve(scale);
			// endregion
			
			agents = newGeneration;
//			System.out.println("Post assign");
//			System.out.println("Agents: " + Arrays.toString(agents));
			
//			System.arraycopy(newGeneration, 0, agents, 0, agents.length);
		}
		
		float percent = ((float) bestScore.get() / outputs.length) * 100;
		String formatted = new DecimalFormat("#.##").format(percent);
		System.out.println("Best: " + formatted + "%");
	}
	
	private int randomIndex(int bound) {
		Random random = new Random();
		return random.nextInt(bound);
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
			
			//calculatedOutputs.forEach(System.out::println);
			
			Optional<Float> max = calculatedOutputs.stream().max(Float::compareTo);
			if (max.isEmpty())
				continue;
			
			if (outputs[i] == calculatedOutputs.indexOf(max.get()))
				score++;
		}
		
		//network.LogWeights();
		System.out.println("Score: " + score);
		return score;
	}
}
