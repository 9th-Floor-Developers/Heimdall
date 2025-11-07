package numbers;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

/**
 * Class representing all operations and processes relating to converting an image of a number into a NumberImage.
 */
public class NumberGenerator {
	/**
	 * Converts all image pixels to array of floats. Floats are calculated by
	 * converting RBG value into a singular greyscale value.
	 *
	 * @param imageFile file to get pixels values
	 * @return 2D float array representing all greyscale values for each pixel in image
	 * @throws Exception if error when reading imageFile using ImageIO
	 */
	private static float[][] imgToFloatArr(File imageFile) throws Exception {
		System.out.print("\rParsing Image: " + imageFile.getName() + " - " + Thread.currentThread().getName());
		
		BufferedImage image = ImageIO.read(imageFile);
		int width = image.getWidth();
		int height = image.getHeight();
		float[][] floatArray = new float[height][width];
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = image.getRGB(x, y);
				int grey = (pixel >> 16) & 0xFF; // extract greyscale value using red component
				floatArray[y][x] = grey / 255.0f; // normalize in 0-1 range
			}
		}
		
		return floatArray;
	}
	
	/**
	 * Verifies that dataset directory exists and is a directory, begins recursive search through directory
	 *
	 * @return array of NumberImages, representing all number images
	 * located in ./src/numbers/dataset and all subdirectories
	 * @throws Exception various errors are thrown based on file status
	 *                   (i.e.: not found, is directory, empty directory, etc.)
	 */
	public static NumberImage[] getAllImgs() throws Exception {
		File dir = new File("./src/numbers/dataset");
		
		if (!dir.exists())
			throw new FileNotFoundException();
		else if (!dir.isDirectory())
			throw new Exception("Selected path is not a directory.");
		
		File[] files = dir.listFiles();
		if (files == null)
			throw new FileNotFoundException("Directory does not contain files.");
		
		ArrayList<NumberImage> arrListVals = searchDir(files);
		System.out.println();
		
		NumberImage[] arrVals = new NumberImage[arrListVals.size()];
		for (int i = 0; i < arrListVals.size(); i++)
			arrVals[i] = arrListVals.get(i);
		
		return arrVals;
	}
	
	public static NumberImage getImg() throws Exception {
		File image = new File("./src/numbers/dataset/0/Zero_full (1).jpg");
		
		if (!image.exists())
			throw new FileNotFoundException();
		else if (image.isDirectory())
			throw new Exception("Selected path is a directory.");
		
		NumberImage numberImage = new NumberImage(imgToFloatArr(image),
		                                          Integer.parseInt(image.getParentFile().getName()));
		System.out.println();
		
		return numberImage;
	}
	
	/**
	 * Recursively and asynchronously searches through all directories and creates NumberImage objects based on
	 * greyscale pixel values of image (pixels[][]) and actual value of image obtained from folder name (value).
	 * <p>
	 * Program assigns threads to subdirectories, parsing images asynchronously, speeding up the parsing process.
	 *
	 * @param files all files in current directory, origin of recursive process
	 * @return ArrayList of NumberImage objects representing all
	 * number images located in directory and all subdirectories
	 * @throws Exception if a problem occurs when converting image to float array
	 */
	private static ArrayList<NumberImage> searchDir(File[] files) throws Exception {
		ArrayList<NumberImage> allImgDecVals = new ArrayList<>();
		ArrayList<Thread> threads = new ArrayList<>();
		
		for (File file : files) {
			if (file.isFile()) {  // image file
				float[][] pixels = imgToFloatArr(file);
				String parent = file.getParentFile().getName();
				NumberImage image = new NumberImage(pixels, Integer.parseInt(parent));
				allImgDecVals.add(image);
			} else if (file.isDirectory()) {  // recursively search subdirectories
				File[] subdirectoryFiles = file.listFiles();
				
				if (subdirectoryFiles == null) {
					System.out.println("Empty directory.");
					continue;
				}
				
				Thread thread = new Thread(() -> {
					ArrayList<NumberImage> subDirImgs;
					try {
						subDirImgs = searchDir(subdirectoryFiles);
					} catch (Exception e) {
						throw new RuntimeException(e);
					}
					allImgDecVals.addAll(subDirImgs);
				});
				
				thread.start();
				threads.add(thread);
			}
		}
		
		for (Thread thread : threads)
			thread.join();
		
		return allImgDecVals;
	}
	
	/**
	 * Simple entrypoint for NumberGenerator.java
	 */
	public static void main(String[] args) throws Exception {
		NumberImage[] allImgDecVals = getAllImgs();
		for (NumberImage image : allImgDecVals) {
			image.printASCII();
			System.out.println("\n==============================================================================\n");
		}
	}
}
