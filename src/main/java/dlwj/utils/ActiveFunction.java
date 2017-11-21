package dlwj.utils;

/**
 *<ul>
 *<li>Description: 激活函数
 *<li>Company: HUST
 *<li>@author Sonly
 *<li>Date: 2017年11月20日
 *</ul>
 */
public final class ActiveFunction {
	
	/**
	 * 阶跃函数
	 * @param x	输入
	 * @return	阶跃函数关于x的值
	 */
	public static int step(double x) {
		if(x >= 0)
			return 1;
		else
			return -1;
	}
	
	/**
	 * sigmoid函数(用于二类逻辑回归)
	 * @param x	输入
	 * @return	sigmoid函数关于x的值
	 */
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
	
	/**
	 * softmax函数(用于多类逻辑回归)
	 * @param x	输入
	 * @param n	类别数
	 * @return	类别的概率
	 */
	public static double[] softmax(double[] x, int n) {
		double[] res = new double[n];
		double max = 0.0;
		double sum = 0.0;
		
		for(int i = 0; i < n; i++) {
			if(x[i] > max)
				max = x[i];
		}
		
		for(int i = 0; i < n; i++) {
			res[i] = Math.exp(x[i] - max);
			sum += res[i];
		}

		for(int i = 0; i < n; i++) {
			res[i] /= sum;
		}
		
		return res;
	}
	
	/**
	 * sigmoid函数的导函数
	 * @param y	sigmoid函数
	 * @return	sigmoid函数的导函数
	 */
	public static double dsigmoid(double y) {
		return y * (1.0 - y);
	}
	
	/**
	 * 双曲正切函数
	 * @param x	输入
	 * @return	x的双曲正切函数值
	 */
	public static double tanh(double x) {
		return Math.tanh(x);
	}
	
	/**
	 * 双曲正切函数的导函数
	 * @param y	双曲正切函数
	 * @return	双曲正切函数的导函数
	 */
	public static double dtanh(double y) {
		return 1.0 - y * y;
	}
}
