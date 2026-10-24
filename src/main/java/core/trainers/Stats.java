package core.trainers;

/**
 * Result of one {@link GpuTrainer#accumulate(java.util.List, boolean, boolean)} pass.
 *
 * @param correct  number of training points whose highest output matched the target digit
 *                 (CPU: the {@code score} counter in {@code FeedForwardTrainer})
 * @param errorSum sum over all points of the average absolute output error; 0 if the error was not requested
 *                 (CPU: the sum of {@code DataUtils.getAverage(agent.backProp(...))})
 */
record Stats(int correct, float errorSum) {}