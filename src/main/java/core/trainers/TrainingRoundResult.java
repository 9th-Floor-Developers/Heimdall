package core.trainers;

import core.utils.DataUtils;

import java.util.List;

public record TrainingRoundResult(int round, float trainingScorePercent, float testingScorePercent, float errorSum) {
    public void print(){
        System.out.println("=========== Round: " + round + " =============");
        System.out.println("Training: " + DataUtils.formatToPercent(trainingScorePercent));
        System.out.println("Testing: " + DataUtils.formatToPercent(testingScorePercent));
        if (errorSum != -1){
            System.out.println("Error rate: " + errorSum);
        }
        System.out.println("===================================");
    }

    public void printMultiple(List<TrainingRoundResult> trainingRoundResults){
        Float[] trainingScores = trainingRoundResults.stream().map(n -> n.trainingScorePercent).toList().toArray(new Float[0]);
        Float[] testingScores = trainingRoundResults.stream().map(n -> n.testingScorePercent).toList().toArray(new Float[0]);
        Float[] errorRates = trainingRoundResults.stream().map(n -> n.errorSum).toList().toArray(new Float[0]);


        System.out.println("=========== Round: " + trainingRoundResults.getFirst().round + " to " + trainingRoundResults.getLast().round + " =============");
        System.out.println("Training: " + DataUtils.arrayToStats(trainingScores, true));
        System.out.println("Testing: "  + DataUtils.arrayToStats(testingScores, true));
        if (errorSum != -1){
            System.out.println("Error rate: " + DataUtils.arrayToStats(errorRates, false));
        }
        System.out.println("===================================");
    }
}
