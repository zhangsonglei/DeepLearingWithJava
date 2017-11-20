package dlwj.perceptrons;

/**
 *<ul>
 *<li>Description: 感知机 
 *<li>Company: HUST
 *<li>@author Sonly
 *<li>Date: 2017年11月20日
 *</ul>
 */
public class Perceptrons {
	
	public int nIn;		//输入数据的维度
	public double[] w;	//感知机的权重向量
	
	public Perceptrons(int nIn) {
		this.nIn = nIn;
		w = new double[nIn];
	}
	
	
	/**
	 * 
	 * @param x		输入数据
	 * @param t		
	 * @param rate	学习速率
	 * @return
	 */
    public int train(double[] x, int t, double rate){
        int classification = 0;
        double c = 0.0;
        
        for(int i = 0; i < nIn; i++) {
        	c += w[i] * x[i] * t;
        }
        
        //检查数据是否被正确分类
        if(c > 0)
        	classification = 1;
        else {//若分类错误，采用梯度下降方法
        	for(int i = 0; i < nIn; i++) {
        		w[i] += rate * x[i] * t;
        	}
        }
        
        return classification;
    }
    
    /**
     * 给定输入，预测其类别
     * @param x	待预测的输入
     * @return	输入的类别
     */
    public int predict(double[] x) {
    	double preActivation = 0.0;
 
    	for(int i = 0; i < nIn; i++) {
    		preActivation += w[i] * x[i];
    	}
    	
    	return ActiveFunction.step(preActivation);
    }
    
    public static void main(String[] args) {
    	
    }
}
