package exceptions;

/**
 * Exception to throw if file fails to delete.
 */
public class FileNotDeleted extends RuntimeException {
	public FileNotDeleted(String message) {
		super(message);
	}
}
