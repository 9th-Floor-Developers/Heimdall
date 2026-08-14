import core.exceptions.IsDirectoryException;
import numberrecognizer.NumberImage;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import numberrecognizer.NumberImageLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.NotDirectoryException;
import java.util.ArrayList;

class NumberImageLoaderTest {
	/*
	@Test
	void getAllImgs() throws Exception {
		NumberImage[] images = NumberImageLoader.getAllImgs("./src/main/resources/numbers/");
		for (NumberImage image : images)
			assert image != null;
		
		assertThrows(NotDirectoryException.class, () -> NumberImageLoader.getAllImgs("./src/main/java/Heimdall.java"));
		assertThrows(FileNotFoundException.class, () -> NumberImageLoader.getAllImgs("./src/DOESNT_EXIST/"));
	}
	
	@Test
	void getRandomImgs() throws Exception {
		int numImages = 50;
		
		NumberImage[] images = NumberImageLoader.getRandomImgs("./src/main/resources/numbers/", numImages);
		for (NumberImage image : images)
			assert image != null;
		assert images.length == numImages;
		
		images = NumberImageLoader.getRandomImgs(images, numImages / 2);
		for (NumberImage image : images)
			assert image != null;
		assert images.length == numImages / 2;
	}
	
	@Test
	void getImg() throws Exception {
		NumberImageLoader.getImg("./src/main/resources/numbers/0/Zero_full (1).jpg");
		
		assertThrows(FileNotFoundException.class, () -> NumberImageLoader.getImg("./src/DOESNT_EXIST.png"));
		assertThrows(IsDirectoryException.class, () -> NumberImageLoader.getImg("./src/main/resources/numbers/"));
	}
	
	@Test
	void imgToFloatArr() throws Exception {
		float[][] greyscaleVals = (float[][]) TestingUtils.invokePrivate(
				NumberImageLoader.class, "imgToFloatArr",
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
		List<NumberImage> images = (List<NumberImage>) TestingUtils.invokePrivate(
				NumberImageLoader.class, "searchDir",
				new Class[]{ File.class },
				new File("./src/main/resources/numbers/0/")
		);
		
		for (NumberImage image : images)
			assert image != null;
	}
	 */
}
