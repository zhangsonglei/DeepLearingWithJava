package dlwj.perceptrons;

/**
 *<ul>
 *<li>Description:  阶跃函数/激活函数
 *<li>Company: HUST
 *<li>@author Sonly
 *<li>Date: 2017年11月20日
 *</ul>
 */
public final class ActiveFunction {
	
	/**
	 * 给定输入，若该输入为负值，返回-1，否则返回1
	 * @param x	输入
	 * @return	-1/1
	 */
	public static int step(double x) {
		if(x >= 0)
			return 1;
		else
			return -1;
	}
}
