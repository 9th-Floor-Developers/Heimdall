package core.trainers;

import core.utils.DataUtils;

public record TrainingRoundResult(int round, float trainingScorePercent, float testingScorePercent, float errorSum) {
    public void print(){
        System.out.println("=========== Round: " + round + " =============");
        System.out.println("Training: " + DataUtils.formatToPercent(trainingScorePercent) + "%");
        System.out.println("Testing: " + DataUtils.formatToPercent(testingScorePercent) + "%");
        if (errorSum != -1){
            System.out.println("Error rate: " + errorSum + "%");
        }
        System.out.println("===================================");
    }
}
