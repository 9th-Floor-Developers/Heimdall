import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class BasicTrainer {
	public void train(float[][] inputs, int[] outputs, int[] layers,
	                  int agents_per_round, int number_of_rounds) throws InterruptedException {
		ArrayList<NeuralNetwork> agents = new ArrayList<>();
		
		for (int i = 0; i < agents_per_round; i++)
			agents.add(new NeuralNetwork(layers));  // new network with two layers of two nodes
		
		int max_score = outputs.length;
		System.out.println("Starting Training");
		
		AtomicInteger best_score = new AtomicInteger(-1);
		AtomicReference<NeuralNetwork> best_agent = new AtomicReference<>(agents.getFirst());
		ArrayList<Thread> threads = new ArrayList<>();
		for (int i = 1; i <= number_of_rounds; i++) {
			for (NeuralNetwork agent : agents) {
				Thread thread = new Thread(() -> {
					int score = eval(agent, inputs, outputs);
					
					if (score > best_score.get()) {
						best_score.set(score);
						best_agent.set(agent);
					}
				});
				thread.start();
				threads.add(thread);
			}
			
			for (Thread thread : threads)
				thread.join();
			threads.clear();
			
			float percent = ((float) best_score.get() / max_score) * 100;
			System.out.println("Round: " + i + " Best score: " + best_score + " Which is: " + percent + "%");
			
			agents = new ArrayList<>();
			for (int j = 0; j < agents_per_round; j++)
				agents.add(best_agent.get().evolve(0.2f));
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
