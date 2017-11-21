package dlwj.perceptrons;

import dlwj.utils.ActiveFunction;
import junit.framework.TestCase;

/**
 *<ul>
 *<li>Description: 测试阶跃函数 
 *<li>Company: HUST
 *<li>@author Sonly
 *<li>Date: 2017年11月21日
 *</ul>
 */
public class ActiveFunctionTest extends TestCase {
	
	protected void setUp() throws Exception {
		super.setUp();
	}

	public void testStep() {
		assertEquals(-1, ActiveFunction.step(-1));
		assertEquals(1, ActiveFunction.step(0));
		assertEquals(1, ActiveFunction.step(1));
	}

}
