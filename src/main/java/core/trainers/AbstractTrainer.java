package core.trainers;

import core.data.DataSet;
import org.jetbrains.annotations.Nullable;
import core.utils.DataLogger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractTrainer {
    @Nullable
    protected DataLogger logger;
    protected int printPerRoundAmount = 1;
    protected int testPerRoundAmount = 1;

    protected List<TrainingRoundResult> trainingRoundResults = new ArrayList<>();

    public AbstractTrainer setPrintPerRoundAmount(int printPerRoundAmount) {
        if (printPerRoundAmount < 1){
            throw new IllegalArgumentException("Print per round amount must be 1 or greater");
        }
        this.printPerRoundAmount = printPerRoundAmount;
        return this;
    }


    public AbstractTrainer setTestPerRoundAmount(int testPerRoundAmount) {
        if (testPerRoundAmount < 1){
            throw new IllegalArgumentException("Test per round amount must be 1 or greater");
        }
        this.testPerRoundAmount = testPerRoundAmount;
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


    public void printTrainingResults(int round){
        if (printPerRoundAmount == 1){
            trainingRoundResults.getLast().print();
        }
        else if (round % printPerRoundAmount == 0){
            List<TrainingRoundResult> section = trainingRoundResults.subList(trainingRoundResults.size() - printPerRoundAmount, trainingRoundResults.size());
            trainingRoundResults.getLast().printMultiple(section);
        }
    }
}
