import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class BasicTrainer {
	float[][] inputs = new float[][]{new float[]{0, 0}, new float[]{1, 0}, new float[]{0, 1}, new float[]{1, 1}};
	int[] outputIndexes = new int[]{0, 1, 1, 1};
	
	List<NeuralNetwork> agents = new ArrayList<>();
	
	
	public void train() {
		int agent_per_round = 100;
		int number_of_rounds = 400;
		
		for (int i = 0; i < agent_per_round; i++){
			agents.add(new NeuralNetwork(new int[]{2, 2}));
		}
		
		System.out.println("Starting Training");
		
		for (int i = 0; i < number_of_rounds; i++){
			int best_score = -1;
			NeuralNetwork best_agent = null;
			
			for (NeuralNetwork agent : agents){
				int score = eval(agent);
				
				if (score > best_score){
					best_score = score;
					best_agent = agent;
				}
			}
			
			System.out.println("Round " + i + " Best score " + best_score);
			
			agents = new ArrayList<>();
			for (int j = 0; j < agent_per_round; j++){
				assert best_agent != null;
				
				agents.add(best_agent.evolve(0.2f));
			}
		}
	}
	
	
	public int eval(NeuralNetwork network) {
		int score = 0;
		for (int i = 0; i < inputs.length; i++) {
			List<Float> calculatedOutputs = network.calculate(inputs[i]);
			
			Optional<Float> max = calculatedOutputs.stream().max(Float::compareTo);
			if (max.isEmpty()){
				continue;
			}
			
			if (outputIndexes[i] == calculatedOutputs.indexOf(max.get())){
				score++;
			}
		}
		
		return score;
	}
}
