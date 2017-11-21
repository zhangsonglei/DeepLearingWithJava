package dlwj.multilayerperceptrons;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import dlwj.logisticRegression.LogisticRegression;

public class MultiLayerPerceptrons {
	
	public int nIn;
	public int nHidden;
	public int nOut;
	public HiddenLayer hiddenLayer;
	public LogisticRegression logisticLayer;
	public Random rng;
	
	public MultiLayerPerceptrons(int nIn, int nHidden, int nOut, Random rng) {
		this.nIn = nIn;
		this.nHidden = nHidden;
		this.nOut = nOut;
		
		if(rng == null)
			rng = new Random(1234);
		
		this.rng = rng;
		
		//使用双曲正切函数作为激活函数构造隐藏层
		hiddenLayer = new HiddenLayer(nIn, nOut, null, null, rng, "tanh");
		
		//使用多类逻辑回归作为输出层
		logisticLayer = new LogisticRegression(nIn, nOut);
	}
	
	public void train(double[][] X, int[][] T, int minibatchSize, double rate) {
		double[][] Z = new double[minibatchSize][nIn];	//隐藏层输出，输出层输入
		double[][] dY;
		
		//前向传播隐藏层
		for(int n = 0; n < minibatchSize; n++) {
			Z[n] = hiddenLayer.forward(X[n]);	//激活输入单元
		}
		
		//前向后向输出层
		dY = logisticLayer.train(Z, T, minibatchSize, rate);
		//后向隐藏层
		hiddenLayer.backward(X, Z, dY, logisticLayer.weight, minibatchSize, rate);		
	}
	
	public Integer[] predict(double[] x) {
		double[] z = hiddenLayer.output(x);
		return logisticLayer.predict(z);
	}
	
	public static void main(String[] args) {
		final Random rng = new Random(1234);
		
		final int patterns = 2;
		final int train_N = 4;
		final int test_N = 4;
		final int nIn = 2;
		final int nHidden = 3;
		final int nOut = patterns;
		
		double[][] train_X;
		int[][] train_T;
		double[][] test_X;
		Integer[][] test_T;
		Integer[][] predict_T = new Integer[test_N][nOut];
	
		final int epochs = 5000;
		double rate = 0.1;
		final int minibatchSize = 1;
		int minibatch_N = train_N / minibatchSize;
		
		double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];
        int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) 
        	minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);

        //准备数据集
        train_X = new double[][]{
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        train_T = new int[][]{
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
        };
        test_X = new double[][]{
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        test_T = new Integer[][]{
                {0, 1},
                {1, 0},
                {1, 0},
                {0, 1}
        };

        //创建小批量数据
        for (int i = 0; i < minibatch_N; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
                train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
            }
        }

        //构造多层感知器模型
        MultiLayerPerceptrons classifier = new MultiLayerPerceptrons(nIn, nHidden, nOut, rng);

        //训练模型
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, rate);
            }
        }

        //预测测试数据
        for (int i = 0; i < test_N; i++) {
            predict_T[i] = classifier.predict(test_X[i]);
        }
        
        //评估模型
        int[][] confusionMatrix = new int[patterns][patterns];
        double accuracy = 0.;
        double[] precision = new double[patterns];
        double[] recall = new double[patterns];

        for (int i = 0; i < test_N; i++) {
            int predicted_ = Arrays.asList(predict_T[i]).indexOf(1);
            int actual_ = Arrays.asList(test_T[i]).indexOf(1);

            confusionMatrix[actual_][predicted_] += 1;
        }

        for (int i = 0; i < patterns; i++) {
            double col_ = 0.;
            double row_ = 0.;

            for (int j = 0; j < patterns; j++) {

                if (i == j) {
                    accuracy += confusionMatrix[i][j];
                    precision[i] += confusionMatrix[j][i];
                    recall[i] += confusionMatrix[i][j];
                }

                col_ += confusionMatrix[j][i];
                row_ += confusionMatrix[i][j];
            }
            precision[i] /= col_;
            recall[i] /= row_;
        }

        accuracy /= test_N;

        System.out.printf("Accuracy = " + accuracy);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) {
            System.out.println("\tclass" + (i + 1) +" = " + precision[i]);
        }
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) {
        	System.out.println("\tclass" + (i + 1) +" = " + recall[i]);
        }

    }
}
