package numbers;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class NumberGenerator {
	private static float[][] imgToFloatArr(File imageFile) throws Exception {
		System.out.print("\rParsing Image: " + imageFile.getName());
		
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
	
	private static NumberImage[] getAllImgs() throws Exception {
		File dir = new File("./src/numbers/dataset");
		
		if (!dir.exists())
			throw new FileNotFoundException();
		else if (!dir.isDirectory())
			throw new Exception("Selected path is not a directory.");
		
		File[] files = dir.listFiles();
		if (files == null)
			throw new FileNotFoundException("Directory does not contain files.");
		
		ArrayList<NumberImage> arrListVals = searchDir(files);
		
		NumberImage[] arrVals = new NumberImage[arrListVals.size()];
		for (int i = 0; i < arrListVals.size(); i++)
			arrVals[i] = arrListVals.get(i);
		
		return arrVals;
	}
	
	private static ArrayList<NumberImage> searchDir(File[] files) throws Exception {
		ArrayList<NumberImage> allImgDecVals = new ArrayList<>();
		
		for (File file : files) {
			if (file.isFile()) {
				float[][] pixels = imgToFloatArr(file);
				String parent = file.getParentFile().getName();
				NumberImage image = new NumberImage(pixels, Integer.parseInt(parent));
				allImgDecVals.add(image);
			} else if (file.isDirectory()) {
				File[] subdirectoryFiles = file.listFiles();
				
				if (subdirectoryFiles == null) {
					System.out.println("Empty directory.");
					continue;
				}
				
				ArrayList<NumberImage> subDirImgs = searchDir(subdirectoryFiles);
				allImgDecVals.addAll(subDirImgs);
			}
		}
		
		return allImgDecVals;
	}
	
	public static void main(String[] args) throws Exception {
		NumberImage[] allImgDecVals = getAllImgs();
		for (NumberImage image : allImgDecVals) {
			image.printASCII();
			System.out.println("\n==============================================================================\n");
		}
	}
}
