package exceptions;

/**
 * Exception to throw if a {@link java.io.File} is a directory and not a file.
 */
public class IsDirectoryException extends RuntimeException {
	public IsDirectoryException(String message) {
		super(message);
	}
}
