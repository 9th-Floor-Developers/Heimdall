package utils;

import exceptions.IsDirectoryException;
import model.data.NumberImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.NotDirectoryException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.DataFormatException;

/**
 * A class containing all operations and processes relating to converting an image of a number into a {@link NumberImage}.
 */
public class NumberUtils {
	/**
	 * Converts all image pixels to array of floats. Floats are calculated by
	 * converting RBG value into a single greyscale value.
	 *
	 * @param imageFile file to get pixels values
	 * @return 2D float array representing all greyscale values for each pixel in image
	 * @throws Exception if error when reading imageFile using {@link ImageIO}
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
				int grey = (pixel >> 16) & 0xFF;  // extract greyscale value using red component
				floatArray[y][x] = 1 - grey / 255.0f;  // normalize in 0-1 range, 1 - ... inverts colors
			}
		}
		
		return floatArray;
	}
	
	/**
	 * Recursively searches through directory and parses all image files after
	 * verifying that the dataset directory exists and is a directory.
	 *
	 * @param src source directory to being recursive search
	 * @return array of {@link NumberImage} objects, representing all number images
	 * located in {@code src} and all subdirectories
	 * @throws Exception various errors are thrown based on file status
	 *                   (i.e.: not found, is directory, empty directory, etc.)
	 */
	public static NumberImage[] getAllImgs(String src) throws Exception {
		File dir = new File(src);
		
		if (!dir.exists())
			throw new FileNotFoundException();
		else if (!dir.isDirectory())
			throw new NotDirectoryException("Selected path is not a directory.");
		
		ArrayList<NumberImage> arrListVals = searchDir(dir);
		System.out.println("\rImage Parsing Complete...");
		
		NumberImage[] arrVals = new NumberImage[arrListVals.size()];
		for (int i = 0; i < arrListVals.size(); i++){
            arrVals[i] = arrListVals.get(i);
            if (arrVals[i].value() == -1){
                throw new DataFormatException("Number image could not be loaded properly, most likely due to invalid folder name ");
            }
        }
		
		return arrVals;
	}
	
	/**
	 * Returns random images from the {@code src} directory and all subdirectories.
	 * <p>
	 * Uses {@link #getAllImgs(String)} to get all images,
	 * then returns {@code numImages} number of images.
	 *
	 * @param src       directory to begin recursive search
	 * @param numImages desired number of {@link NumberImage} objects.
	 * @return {@link NumberImage} array representing {@code numImages}
	 * random images from {@code src} directory and subdirectories.
	 * @throws Exception if directory is not found, empty, or is not a directory.
	 */
	public static NumberImage[] getRandomImgs(String src, int numImages, int seed) throws Exception {
		NumberImage[] allImages = getAllImgs(src),
				randomImages = new NumberImage[numImages];
		
		Random random = new Random(seed);
		for (int i = 0; i < numImages; i++)
			randomImages[i] = allImages[random.nextInt(allImages.length)];
		
		return randomImages;
	}
	
	/**
	 * Returns random images from {@code allImages} {@link NumberImage} selection array.
	 * <p>
	 * A smaller selection of images will increase the number of duplicate {@link NumberImage} objects.
	 *
	 * @param allImages {@link NumberImage} array to pick images from.
	 * @param numImages desired number of {@link NumberImage} objects.
	 * @return {@link NumberImage} array representing {@code numImages}
	 * random images from {@code allImages} array.
	 */
	public static NumberImage[] getRandomImgs(NumberImage[] allImages, int numImages, int seed) {
		NumberImage[] randomImages = new NumberImage[numImages];
		
		Random random = new Random(seed);
		for (int i = 0; i < numImages; i++)
			randomImages[i] = allImages[random.nextInt(allImages.length)];
		
		return randomImages;
	}
	
	/**
	 * Parses single specified image into {@link NumberImage}
	 *
	 * @param src source path of target image
	 * @return image as {@link NumberImage} object
	 * @throws Exception if file at src location does not exist or is a directory
	 */
	@SuppressWarnings("UnusedReturnValue")
    public static NumberImage getImg(String src) throws Exception {
		File image = new File(src);
		
		if (!image.exists())
			throw new FileNotFoundException();
		else if (image.isDirectory())
			throw new IsDirectoryException("Selected path is a directory.");
		
		NumberImage numberImage = new NumberImage(imgToFloatArr(image),
		                                          Integer.parseInt(image.getParentFile().getName()));
		System.out.println("\rImage Parsing Complete...");

		return numberImage;
	}
	
	/**
	 * Recursively and asynchronously searches through all directories and
	 * creates {@link NumberImage} objects based on greyscale pixel values
	 * of image ({@code pixels[][]}) and actual value of image obtained
	 * from folder name ({@code value}).
	 * <p>
	 * Program assigns a {@link Thread} to subdirectories, parsing images asynchronously, speeding up the parsing process.
	 *
	 * @param dir directory to check files in as a {@link File} object, origin of recursive process
	 * @return ArrayList of {@link NumberImage} objects representing all
	 * number images located in directory and all subdirectories
     * any {@link NumberImage} without a numeric folder name will have -1 as its value
     *
	 * @throws Exception if a problem occurs when converting image to float array
	 */
	private static ArrayList<NumberImage> searchDir(File dir) throws Exception {
		ArrayList<NumberImage> allImgs = new ArrayList<>();
        ArrayList<ArrayList<NumberImage>> allSublists = new ArrayList<>();

		ArrayList<Thread> threads = new ArrayList<>();

		File[] files = dir.listFiles();
		if (files == null)
			throw new FileNotFoundException("Directory does not contain files.");

        int value = -1;
        try {
            value = Integer.parseInt(dir.getName());
        }
        catch (Exception ignored){}

        final int dirValue = value;

		for (File file : files) {
			if (!file.isDirectory()) {  // file
				float[][] pixels = imgToFloatArr(file);
				NumberImage image = new NumberImage(pixels, dirValue);
				allImgs.add(image);
				continue;
			}
			
			// asynchronously and recursively search subdirectories
			File[] subdirectoryFiles = file.listFiles();
			
			if (subdirectoryFiles == null) {
				System.out.println("Empty directory: " + file.getName());
				continue;
			}

            ArrayList<NumberImage> newSubList = new ArrayList<>();
			
			Thread thread = new Thread(() -> {
				ArrayList<NumberImage> subDirImgs;
				try {
					subDirImgs = searchDir(file);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
                newSubList.addAll(subDirImgs);
			});

			allSublists.add(newSubList);
			threads.add(thread);
			thread.start();
		}

		for (Thread thread : threads)
            thread.join();

        allSublists.forEach(allImgs::addAll);
		
		return allImgs;
	}
}
