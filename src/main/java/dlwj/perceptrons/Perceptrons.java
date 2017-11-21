package dlwj.perceptrons;

import static dlwj.utils.ActiveFunction.step;
import java.util.Random;
import dlwj.utils.GaussianDistribution;

/**
 *<ul>
 *<li>Description: 感知器
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
	 * 训练模型
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
    	
    	return step(preActivation);
    }
    
    public static void main(String[] args) {
    	final int train_N 	= 1000;	//训练数据的数量
    	final int test_N 	= 200;	//测试数据的数量
    	final int nIn 		= 2;	//输入数据的维度
    	
    	double[][] train_x = new double[train_N][nIn];	//用于训练的输入数据
    	int[]      train_T = new int[train_N];			//用于训练的输出数据
    	
    	double[][] test_x = new double[test_N][nIn];	//用于测试的输入数据
    	int[]      test_T = new int[test_N];			//用于测试的实际标记
    	int[]   predict_T = new int[test_N];			//模型预测的输出
    	
    	final int epochs 	= 2000;	//最大迭代次数
    	final double rate	= 1.0;	//学习率
    	
    	final Random rng = new Random(1234);
    	GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
    	GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
    	
    	//随机得到服从高斯分布的数据集
    	for(int i = 0; i < train_N / 2 - 1; i++) {
    		train_x[i][0] = g1.random();
    		train_x[i][1] = g2.random();
    		train_T[i] 	  = 1;
    	}
    	for(int i = 0; i < test_N / 2 - 1; i++) {
    		test_x[i][0] = g1.random();
    		test_x[i][1] = g2.random();
    		test_T[i] 	  = 1;
    	}
    	
    	for(int i = train_N / 2; i < train_N; i++) {
    		train_x[i][0] = g2.random();
    		train_x[i][1] = g1.random();
    		train_T[i] 	  = -1;
    	}
    	for(int i = test_N / 2; i < test_N; i++) {
    		test_x[i][0] = g2.random();
    		test_x[i][1] = g1.random();
    		test_T[i] 	  = -1;
    	}
    	
    	int epoch = 0;
    	Perceptrons classifier = new Perceptrons(nIn);
    	
    	//训练模型
    	while(true) {
    		int classified = 0;
    		
    		for(int i = 0; i < train_N; i++) {
    			classified += classifier.train(train_x[i], train_T[i], rate);
    		}
    		
    		if(classified == train_N)
    			break;
    		
    		epoch++;
    		if(epoch > epochs)
    			break;
    	}
    	
    	//模型预测
    	for(int i = 0; i < test_N; i++) {
    		predict_T[i] = classifier.predict(test_x[i]);
    	}
    	
    	/**
    	 * 模型评价:
    	 * Accuracy  = (TP+TN)/(TP+TN+FP+FN)
    	 * Precision = TP/(TP+FP)
    	 * Recall    = TP/(TP+FN)
    	 ****************************
    	 *         Ptrue  *  Pfalse *
    	 ****************************
    	 *Atrue  *  TP   *    FN    *
    	 ****************************
    	 *Afalse *  FP   *    TN    *
    	 ****************************
    	 */
    	int[][] confusionMatrix = new int[2][2];
    	double Accuracy = 0.0;
    	double Precision = 0.0;
    	double Recall = 0.0;
    	
    	for(int i = 0; i < test_N; i++) {
    		if(predict_T[i] > 0) {
    			if(test_T[i] > 0) {
    				Accuracy += 1;
    				Precision+= 1;
    				Recall   += 1;
    				confusionMatrix[0][0] += 1;
    			}else {
    				confusionMatrix[1][0] += 1;
    			}
    		}else {
    			if(test_T[i] > 0) {
    				confusionMatrix[0][1] += 1;
    			}else {
    				Accuracy += 1;
    				confusionMatrix[1][1] += 1;
    			}
    		}
    	}
    	
    	Accuracy	/= test_N;
    	Precision	/= confusionMatrix[0][0] + confusionMatrix[1][0];
    	Recall		/= confusionMatrix[0][0] + confusionMatrix[0][1];

    	System.out.println("Accuracy = " + Accuracy);
    	System.out.println("Precision = " + Precision);
    	System.out.println("Recall = " + Recall);
    }
}
