package dlwj.utils;

import java.util.Random;

/**
 *<ul>
 *<li>Description: 随机数生成器 
 *<li>Company: HUST
 *<li>@author Sonly
 *<li>Date: 2017年11月21日
 *</ul>
 */
public final class RandomGenerator {

	/**
	 * 基于均匀分布生成随机数
	 * @param min	随机数的最小值
	 * @param max	随机数的最大值
	 * @param rng	随机数函数
	 * @return		[min, max]之间的随机数
	 */
	public static double uniform(double min, double max, Random rng) {
		return rng.nextDouble() * (max - min) + min;
	}
}
