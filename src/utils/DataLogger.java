package utils;

import model.NeuralNetwork;

import java.io.*;

/**
 * Data Utility Class, contains several data management saving utility classes.
 *
 * @see #log(int, float, int, String)
 * @see #saveAgent(NeuralNetwork)
 * @see #loadAgent(String)
 */
public class DataLogger {
	private final String path;
	private File data;
	
	/**
	 * Creates datalogger object, giving access to its methods.
	 *
	 * @param path folder path to store data in;
	 *             child folders will be made for better organization
	 * @see #log(int, float, int, String)
	 * @see #saveAgent(NeuralNetwork)
	 * @see #loadAgent(String)
	 */
	public DataLogger(String path) {
		File folder = new File(path);
		this.path = path + "/" + findNextSessionNumber(folder);
	}
	
	/**
	 * Initializes data logging, to {@code selectedFolder/subFolder/training-data.csv}.
	 * <p>
	 * {@link #log(int, float, int, String)} can now be called to log training data.
	 *
	 * @throws IOException            if file writer error occurs
	 * @throws InstantiationException if error occurs when creating subfolder
	 */
	public void initLogger() throws IOException, InstantiationException {
		File folder = new File(path);
		if (!folder.mkdir())
			throw new InstantiationException("Failed To Create New Folder");
		
		data = new File(path + "/training-data.csv");
		
		FileWriter writer = new FileWriter(data);
		writer.write("Generation, Score, Total, Percent\n");
		writer.close();
	}
	
	/**
	 * Finds the highest folder number out of subfolder names in {@code trainingPath} folder.
	 *
	 * @param trainingFolder folder as a {@link File} object to check subdirectory names in.
	 * @return 1 higher than the highest folder.
	 */
	private int findNextSessionNumber(File trainingFolder) {
		int max = 0;
		for (String s : trainingFolder.list()) {
			int number = Integer.parseInt(s);
			if (number > max)
				max = number;
		}
		return max + 1;
	}
	
	/**
	 * Logs a generation's data to {@code selectedFolder/subFolder/training-data.csv}.
	 *
	 * @param generationNum generation index
	 * @param score         number of images network correctly identified
	 * @param inputsLength  total number of input images in inputs
	 * @param formatted     percent of images guessed correctly in {@link String} format
	 * @throws IOException if file writing error occurs
	 */
	public void log(int generationNum, float score, int inputsLength, String formatted) throws IOException {
		if (data == null)
			throw new NullPointerException(
					"Logger Has Not Been Initialized, put DataLogger.initLogger(), before DataLogger.log(...).");
		
		FileWriter writer = new FileWriter(data, true);
		writer.write(generationNum + "," + score + "," + inputsLength + "," + formatted + "\n");
		writer.close();
	}
	
	/**
	 * Serializes an agent to a file in {@code selectedFolder/subFolder/agent.ser}.
	 * <p>
	 * Serialized object can be unserialized using {@link #loadAgent(String)}.
	 *
	 * @param agent {@link NeuralNetwork} object to be serialized
	 */
	public void saveAgent(NeuralNetwork agent) {
		try (FileOutputStream fileOut = new FileOutputStream(path + "/agent.ser");
		     ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
			out.writeObject(agent);
			System.out.println("Agent Saved Successfully...");
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * Loads an agent from a serialized file, {@code folder/agent.ser}.
	 * <p>
	 * Objects can be serialized to a file using {@link #saveAgent(NeuralNetwork)}.
	 *
	 * @param folder path of folder that serialized agent object is located in
	 * @return {@link NeuralNetwork} object loaded from a serialized function
	 */
	public NeuralNetwork loadAgent(String folder) {
		try (FileInputStream fileIn = new FileInputStream(folder + "/agent.ser");
		     ObjectInputStream in = new ObjectInputStream(fileIn)) {
			NeuralNetwork agent = (NeuralNetwork) in.readObject();
			System.out.println("Agent Loaded Successfully");
			return agent;
		} catch (IOException | ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
}
