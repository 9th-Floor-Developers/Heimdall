package trainer;

import model.NeuralNetwork;
import org.jetbrains.annotations.Nullable;
import utils.DataLogger;

public abstract class AbstractTrainer {
    @Nullable
    protected DataLogger logger;




    abstract public void trainAgent();

    abstract public float getBestScore();


    /**
     * initializes a {@link DataLogger} object to the current Trainer object.
     *
     * @return current object, allowing for inheritance chain and one-line setup
     * @throws Exception if error occurs in {@link DataLogger#initLogger()}
     */
    public AbstractTrainer addLogger() throws Exception {
        logger = new DataLogger("./src/training-results");
        logger.initLogger();

        return this;
    }

    /**
     * Saves an agent to a serialized object.
     * <p>
     * Agent can be loaded using {@link DataLogger#loadAgent(String)}.
     *
     * @param agentName name of serialized agent file
     */
    abstract public void saveAgent(String agentName);
}
