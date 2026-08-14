package numberrecognizer;

import core.data.AbstractDataSetLoader;
import core.data.DataPoint;
import core.exceptions.IsDirectoryException;


import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.NotDirectoryException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A class containing all operations and processes relating to converting an image of a number into a {@link NumberImage}.
 */
public class NumberImageLoader extends AbstractDataSetLoader {
	private final AtomicInteger loadAmount = new AtomicInteger(0);

	public static NumberImageLoader createLoader(){
		return new NumberImageLoader();
	}

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
	 * @param loadLimit the number of data points that is being requested
	 *
	 * @return list of {@link DataPoint} objects, representing all number images
	 * located in {@code src} and all subdirectories
	 * @throws Exception various errors are thrown based on file status
	 *                   (i.e.: not found, is directory, empty directory, etc.)
	 */

	@Override
	protected List<DataPoint> loadDataPoints(int loadLimit) throws Exception {
		if (src == null)
			throw new IllegalStateException("Source is not specified");

		File dir = new File(src);

		if (!dir.exists())
			throw new FileNotFoundException();
		else if (!dir.isDirectory())
			throw new NotDirectoryException("Selected path is not a directory.");

		List<NumberImage> numberImgs = searchDir(dir);
		System.out.println("\rImage Parsing Complete, for " + numberImgs.size() + " images");

		return numberImgs.stream().map(n -> (DataPoint) n).toList();
	}
	
	/**
	 * Parses single specified image into {@link NumberImage}
	 *
	 * @param src source path of target image
	 * @return image as {@link NumberImage} object
	 * @throws Exception if file at src location does not exist or is a directory
	 */
	@SuppressWarnings("unused")
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
	private List<NumberImage> searchDir(File dir) throws Exception {
		List<NumberImage> allImgs = new ArrayList<>();
        List<List<NumberImage>> allSublists = new ArrayList<>();

		List<Thread> threads = new ArrayList<>();

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

				if (loadAmount.get() >= loadLimit){
					return allImgs;
				}
				allImgs.add(image);
				loadAmount.incrementAndGet();
				continue;
			}
			
			// asynchronously and recursively search subdirectories
			File[] subdirectoryFiles = file.listFiles();
			
			if (subdirectoryFiles == null) {
				System.out.println("Empty directory: " + file.getName());
				continue;
			}

            List<NumberImage> newSubList = new ArrayList<>();
			
			Thread thread = new Thread(() -> {
				List<NumberImage> subDirImgs;
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
