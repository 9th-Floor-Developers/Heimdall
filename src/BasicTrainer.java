import java.util.ArrayList;
import java.util.Optional;

public class BasicTrainer {
	public void train(float[][] inputs, int[] outputs, int[] layers) {
        ArrayList<NeuralNetwork> agents = new ArrayList<>();
		int agent_per_round = 5;
		int number_of_rounds = 25;
		
		for (int i = 0; i < agent_per_round; i++)
			agents.add(new NeuralNetwork(layers));  // new network with two layers of two nodes
		
		System.out.println("Starting Training");
		
		int best_score = -1;
		NeuralNetwork best_agent = agents.getFirst();
		for (int i = 1; i <= number_of_rounds; i++) {
			for (NeuralNetwork agent : agents) {
				int score = eval(agent, inputs, outputs);
				
				if (score > best_score) {
					best_score = score;
					best_agent = agent;
				}
			}
			
			System.out.println("Round " + i + " Best score " + best_score);
			
			agents = new ArrayList<>();
			for (int j = 0; j < agent_per_round; j++) {
				agents.add(best_agent.evolve(0.2f));
			}
		}
	}
	
	
	public int eval(NeuralNetwork network, float[][] inputs, int[] outputs) {
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
