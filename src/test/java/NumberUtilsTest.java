import exceptions.IsDirectoryException;
import model.data.NumberImage;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import utils.NumberUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.NotDirectoryException;
import java.util.ArrayList;

class NumberUtilsTest {
	@Test
	void getAllImgs() throws Exception {
		NumberImage[] images = NumberUtils.getAllImgs("./src/main/resources/numbers/");
		for (NumberImage image : images)
			assert image != null;
		
		assertThrows(NotDirectoryException.class, () -> NumberUtils.getAllImgs("./src/main/java/Heimdall.java"));
		assertThrows(FileNotFoundException.class, () -> NumberUtils.getAllImgs("./src/DOESNT_EXIST/"));
	}
	
	@Test
	void getRandomImgs() throws Exception {
		int numImages = 50, seed = 123;
		
		NumberImage[] images = NumberUtils.getRandomImgs("./src/main/resources/numbers/", numImages, seed);
		for (NumberImage image : images)
			assert image != null;
		assert images.length == numImages;
		
		images = NumberUtils.getRandomImgs(images, numImages / 2, seed);
		for (NumberImage image : images)
			assert image != null;
		assert images.length == numImages / 2;
	}
	
	@Test
	void getImg() throws Exception {
		NumberUtils.getImg("./src/main/resources/numbers/0/Zero_full (1).jpg");
		
		assertThrows(FileNotFoundException.class, () -> NumberUtils.getImg("./src/DOESNT_EXIST.png"));
		assertThrows(IsDirectoryException.class, () -> NumberUtils.getImg("./src/main/resources/numbers/"));
	}
	
	@Test
	void imgToFloatArr() throws Exception {
		float[][] greyscaleVals = (float[][]) TestingUtils.invokePrivate(
				NumberUtils.class, "imgToFloatArr",
				new Class[] { File.class },
				new File("./src/main/resources/numbers/0/Zero_full (1).jpg")
		);
		
		for (float[] greyScaleArr : greyscaleVals)
			for (float v : greyScaleArr)
				assert v >= 0 && v <= 1;
	}
	
	@Test
	void searchDir() throws Exception {
		//noinspection unchecked
		ArrayList<NumberImage> images = (ArrayList<NumberImage>) TestingUtils.invokePrivate(
				NumberUtils.class, "searchDir",
				new Class[]{ File.class },
				new File("./src/main/resources/numbers/0/")
		);
		
		for (NumberImage image : images)
			assert image != null;
	}
}
