import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class BasicTrainer {
	public void train(float[][] inputs, int[] outputs, int[] layers,
	                  int agentsPerRound, int numRounds) throws InterruptedException {
		ArrayList<NeuralNetwork> agents = new ArrayList<>();
		
		for (int i = 0; i < agentsPerRound; i++)
			agents.add(new NeuralNetwork(layers));  // new network with layer structure as specified in layers variable
		
		int maxScore = outputs.length;
		System.out.println("Training...");
		
		AtomicInteger bestScore = new AtomicInteger(-1);
		AtomicReference<NeuralNetwork> bestAgent = new AtomicReference<>(agents.getFirst());
		ArrayList<Thread> threads = new ArrayList<>();
		for (int i = 1; i <= numRounds; i++) {
			for (NeuralNetwork agent : agents) {
				Thread thread = new Thread(() -> {
					int score = eval(agent, inputs, outputs);
					
					if (score > bestScore.get()) {
						bestScore.set(score);
						bestAgent.set(agent);
					}
				});
				thread.start();
				threads.add(thread);
			}
			
			for (Thread thread : threads)
				thread.join();
			threads.clear();
			
			float percent = ((float) bestScore.get() / maxScore) * 100;
			System.out.println("Round: " + i + " Best score: " + bestScore + " Which is: " + percent + "%");
			
			agents = new ArrayList<>();
			for (int j = 0; j < agentsPerRound; j++)
				agents.add(bestAgent.get().evolve(.5f));
		}
	}
	
	
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
