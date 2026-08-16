package core.trainers;

import core.data.DataSet;
import org.jetbrains.annotations.Nullable;
import core.utils.DataLogger;

import java.io.IOException;

public abstract class AbstractTrainer {
    @Nullable
    protected DataLogger logger;
    protected int printPerRoundAmount = 1;

    public AbstractTrainer setPrintPerRoundAmount(int printPerRoundAmount) {
        if (printPerRoundAmount < 1){
            throw new IllegalArgumentException("Print per round amount must be 1 or greater");
        }
        this.printPerRoundAmount = printPerRoundAmount;
        return this;
    }

    abstract public void trainAgent(DataSet dataSet) throws IOException;


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
}
