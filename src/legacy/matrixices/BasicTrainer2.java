package legacy.matrixices;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class BasicTrainer2 {
	public void train(float[][] inputs, int[] outputs, int[] layers,
	                  int agentsPerRound, int numRounds, float scale) throws InterruptedException {
		ArrayList<NeuralNetwork> agents = new ArrayList<>();
		
		for (int i = 0; i < agentsPerRound; i++)
			agents.add(new NeuralNetwork(layers));  // new network with layer structure as specified in layers variable
		
		int maxScore = outputs.length;
		System.out.println("Training...");
		
		AtomicInteger bestScore = new AtomicInteger(-1);
		AtomicReference<NeuralNetwork> bestAgent = new AtomicReference<>(agents.getFirst());
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
			
			float percent = ((float) roundBest.get() / maxScore) * 100;
			String formatted = new DecimalFormat("#.##").format(percent);
			
			System.out.println("Round: " + i + " Best score: " + roundBest.get() + " Which is: " + formatted + "%");
			
			agents = new ArrayList<>();
			for (int j = 0; j < agentsPerRound; j++)
				agents.add(bestAgent.get().evolve_all(scale));
		}
		
		float percent = ((float) bestScore.get() / maxScore) * 100;
		String formatted = new DecimalFormat("#.##").format(percent);
		System.out.println("Best Score: " + percent + "%");
	}
	
	
	private int eval(NeuralNetwork network, float[][] inputs, int[] outputs) {
		int score = 0;
		for (int i = 0; i < inputs.length; i++) {
			float[] calculatedOutputs = network.calculate(inputs[i]);
			
			int maxIndex = 0;
			float max = calculatedOutputs[0];
			for (int j = 0; j < calculatedOutputs.length; j++){
				if (calculatedOutputs[j] > max){
					max = calculatedOutputs[j];
					maxIndex = j;
				}
			}
			
			if (outputs[i] == maxIndex)
				score++;
		}
		
		return score;
	}
}
