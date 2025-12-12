import model.NeuralNetwork;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NeuralNetworkTest {
	private NeuralNetwork network;
	
	@BeforeEach
	void setUp() {
		int[] layerLengths = {
				500,
				100,
				10
		};
		
		network = new NeuralNetwork(layerLengths, 123);
	}
	
	@Test
	void calcOutputs() {
	}
	
	@Test
	void backProp() {
	}
	
	@Test
	void applyWeights() {
	}
	
	@Test
	void getLayers() {
	}
	
	@Test
	void getLayerLengths() {
	}
	
	@Test
	void getLayer() {
	}
	
	@Test
	void getNeuron() {
	}
	
	@Test
	void setNeuron() {
	}
	
	@Test
	void setWeights() {
	}
	
	@Test
	void getWeights() {
	}
	
	@Test
	void setBiases() {
	}
	
	@Test
	void getBiases() {
	}
}
