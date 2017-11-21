package dlwj.logisticRegression;

import static dlwj.utils.ActiveFunction.softmax;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import dlwj.utils.GaussianDistribution;
/**
 *<ul>
 *<li>Description: 逻辑回归 
 *<li>Company: HUST
 *<li>@author Sonly
 *<li>Date: 2017年11月21日
 *</ul>
 */
public class LogisticRegression {
	
	public int nIn;				//输入数据维度
	public int nOut;			//输出数据维度
	public double[][] weight;	//权重向量
	public double[] bias;		//偏差向量
	
	public LogisticRegression(int nIn, int nOut) {
		this.nIn = nIn;
		this.nOut = nOut;
		weight = new double[nOut][nIn];
		bias = new double[nOut];
	}
	
	/**
	 * 
	 * @param x
	 * @return
	 */
	public double[] output(double[] x) {
		double[] preActivition = new double[nOut];
		
		for(int i = 0; i < nOut; i++) {
			for(int j = 0; j < nIn; j++) {
				preActivition[i] += weight[i][j] * x[j];
			}
			preActivition[i] += bias[i];	//线性输出
		}
		
		return softmax(preActivition, nOut);
	}
	
	/**
	 * 训练模型
	 * @param X				训练数据
	 * @param T				
	 * @param minibatchSize	小批量的大小
	 * @param rate			学习速率
	 * @return				返回
	 */
	public double[][] train(double[][] X, int[][] T, int minibatchSize, double rate) {
		double[][] res 			= new double[minibatchSize][nOut];	//结果
		double[][] grad_weight 	= new double[nOut][nIn];			//权重梯度
		double[]   grad_bias 	= new double[nOut];					//偏差梯度
		
		//使用SGD（随机梯度下降）计算梯度
		for(int n = 0; n < minibatchSize; n++) {
			double[] predict_Y = output(X[n]);
			
			for(int i = 0; i < nOut; i++) {
				res[n][i] = predict_Y[i] - T[n][i];
				for(int j = 0; j < nIn; j++) {
					grad_weight[i][j] += res[n][i] * X[n][j];
				}
				
				grad_bias[i] += res[n][i];
			}
		}
		
		//更新梯度
		for(int i = 0; i < nOut; i++) {
			for(int j = 0; j < nIn; j++) {
				weight[i][j] -= rate * grad_weight[i][j] / minibatchSize;
			}
			bias[i] -= rate * grad_bias[i] / minibatchSize;
		}
		return res;
	}
	
	/**
	 * 预测类别
	 * @param x	待预测的数据
	 * @return	预测的结果
	 */
	public Integer[] predict(double[] x) {
		double[] y = output(x);
		Integer[] res = new Integer[nOut];
		
		int maxIndex = -1;
		double max = 0.0;
		
		//最大值的索引
		for(int i = 0; i < nOut; i++) {
			if(max < y[i]) {
				max = y[i];
				maxIndex = i;
			}
		}
		
		//最大值对应的标签为1，其余为0
		for(int i = 0; i < nOut; i++) {
			if(i == maxIndex)
				res[i] = 1;
			else
				res[i] = 0;
		}
		
		return res;
	}
	
	public static void main(String[] args) {		
		//随机数
		Random rng = new Random(1234);
		
		final int patterns = 3;				//类别数
		final int train_N = 400 * patterns;	//训练数据数量
		final int test_N = 60 * patterns;	//测试数据数量
		final int nIn = 2;					//输入数据的维度
		final int nOut = patterns;			//输出数据的维度
		
		double[][] train_X = new double[train_N][nIn];
		int[][] train_T = new int[train_N][nOut];
		
		double[][] test_X = new double[test_N][nIn];
		Integer[][] test_T = new Integer[test_N][nOut];
		Integer[][] predict_T = new Integer[test_N][nOut];
		
		int epochs = 2000;							//最大迭代数量
		double rate = 0.2;							//学习率
		int minibatchSize = 50;						//每个小批量大小
		int minibatch_N = train_N / minibatchSize;	//小批量个数
	
		double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];	//训练数据的批量数组
		int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];		//训练数据的类别批量数组
		
		List<Integer> minibatchIndex = new ArrayList<Integer>();						//用于SGD的数据索引
		for(int i = 0; i < train_N; i++) 
			minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);										//随机打乱数据索引
	
		
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
		GaussianDistribution g3 = new GaussianDistribution(0.0, 1.0, rng);
		
		//类别1数据集
        for (int i = 0; i < train_N/patterns - 1; i++) {
            train_X[i][0] = g1.random();
            train_X[i][1] = g2.random();
            train_T[i] = new int[]{1, 0, 0};
        }
        for (int i = 0; i < test_N/patterns - 1; i++) {
            test_X[i][0] = g1.random();
            test_X[i][1] = g2.random();
            test_T[i] = new Integer[]{1, 0, 0};
        }

        //类别2数据集
        for (int i = train_N/patterns - 1; i < train_N/patterns * 2 - 1; i++) {
            train_X[i][0] = g2.random();
            train_X[i][1] = g1.random();
            train_T[i] = new int[]{0, 1, 0};
        }
        for (int i = test_N/patterns - 1; i < test_N/patterns * 2 - 1; i++) {
            test_X[i][0] = g2.random();
            test_X[i][1] = g1.random();
            test_T[i] = new Integer[]{0, 1, 0};
        }

        //类别3数据集
        for (int i = train_N/patterns * 2 - 1; i < train_N; i++) {
            train_X[i][0] = g3.random();
            train_X[i][1] = g3.random();
            train_T[i] = new int[]{0, 0, 1};
        }
        for (int i = test_N/patterns * 2 - 1; i < test_N; i++) {
            test_X[i][0] = g3.random();
            test_X[i][1] = g3.random();
            test_T[i] = new Integer[]{0, 0, 1};
        }
        
        //使用训练数据创建小批量数据集用于SGD
        for (int i = 0; i < minibatch_N; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
                train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
            }
        }
        
        
        LogisticRegression logisticRegression = new LogisticRegression(nIn, nOut);
        //训练模型
        for(int epoch = 0; epoch < epochs; epoch++) {
        	for(int batch = 0; batch < minibatch_N; batch++) {
        		logisticRegression.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, rate);
        	}
        	rate *= 0.95;
        }
        
        //预测测试数据的类别
        for(int i = 0; i < test_N; i++) {
        	predict_T[i] = logisticRegression.predict(test_X[i]);
        }
        
        //评估模型
        int[][] confusionMatrix = new int[patterns][patterns];
        double accuracy = 0.0;
        double[] precision = new double[patterns];
        double[] recall = new double[patterns];

        for (int i = 0; i < test_N; i++) {
            int predict = Arrays.asList(predict_T[i]).indexOf(1);
            int actual = Arrays.asList(test_T[i]).indexOf(1);

            confusionMatrix[actual][predict] += 1;
        }

        for (int i = 0; i < patterns; i++) {
            double col = 0.;
            double row = 0.;

            for (int j = 0; j < patterns; j++) {
                if (i == j) {
                    accuracy += confusionMatrix[i][j];
                    precision[i] += confusionMatrix[j][i];
                    recall[i] += confusionMatrix[i][j];
                }

                col += confusionMatrix[j][i];
                row += confusionMatrix[i][j];
            }
            precision[i] /= col;
            recall[i] /= row;
        }

        accuracy /= test_N;

        System.out.println("Accuracy = " + accuracy);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) 
            System.out.println("\tclass" + (i + 1) +" = " + precision[i]);
        
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) 
        	System.out.println("\tclass" + (i + 1) +" = " + recall[i]);
	}
}
