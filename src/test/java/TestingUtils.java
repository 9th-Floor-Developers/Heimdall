import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

class TestingUtils {
	static Object invokePrivate(Class<?> clazz, String name, Class<?>[] paramTypes, Object... params) throws Exception {
		Method method = clazz.getDeclaredMethod(name, paramTypes);
		method.setAccessible(true);
		
		Object instance = Modifier.isStatic(method.getModifiers()) ? null : clazz;
		return method.invoke(instance, params);
	}
	
	static Object invokePrivate(Object target, String name, Class<?>[] args, Object... params) throws Exception {
		Method method = target.getClass().getDeclaredMethod(name, args);
		method.setAccessible(true);
		return method.invoke(target, params);
	}
	
	static Object getPrivate(Object target, String name) throws Exception {
		Field field = target.getClass().getDeclaredField(name);
		field.setAccessible(true);
		return field.get(target);
	}
}
