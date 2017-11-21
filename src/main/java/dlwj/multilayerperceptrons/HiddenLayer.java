package dlwj.multilayerperceptrons;

import java.util.Random;
import java.util.function.DoubleFunction;
import static dlwj.utils.RandomGenerator.uniform;
import static dlwj.utils.ActiveFunction.*;

public class HiddenLayer {
	
	public int nIn;
	public int nOut;
	public double[][] weight;
	public double[] bias;
	public Random rng;
	public DoubleFunction<Double> activation;
	public DoubleFunction<Double> dactivation;
	
	public HiddenLayer(int nIn, int nOut, double[][] weight, double[] bias, Random rng, String activation) {
		if(rng == null)
			rng = new Random(1234);
		
		if(weight == null) {
			weight = new double[nOut][nIn];
			
			double w = 1.0 / nIn;
			for(int i = 0; i < nOut; i++) {
				for(int j = 0; j < nIn; j++) {
					weight[i][j] = uniform(-w, w, rng);
				}
			}
		}
		
		if(bias == null) 
			bias = new double[nOut];
				
		this.nIn = nIn;
		this.nOut = nOut;
		this.weight = weight;
		this.bias = bias;
		this.rng = rng;
		
		if(activation == "sigmoid" || activation == null) {
			this.activation = (double x) -> sigmoid(x);
			this.dactivation = (double x) -> dsigmoid(x);
		}else if(activation == "tanh") {
			this.activation = (double x) -> tanh(x);
			this.dactivation = (double x) -> dtanh(x);
		}else
			throw new IllegalArgumentException("不支持 " + activation + " 激活函数");
	}
	
	/**
	 * 
	 * @param x
	 * @return
	 */
	public double[] output(double[] x) {
		double[] res = new double[nOut];
		
		for(int i = 0; i < nOut; i++) {
			double preActivation = 0.0;
			
			for(int j = 0; j < nIn; j++) {
				preActivation += weight[i][j] * x[j];
			}
			preActivation += bias[i];
			res[i] = activation.apply(preActivation);
		}
	
		return res;
	}
	
	/**
	 * 前向传播
	 * @param x
	 * @return
	 */
	public double[] forward(double[] x) {
		return output(x);
	}
	
	public double[][] backward(double[][] X, double[][] Z, double[][] dY,  double[][] Wprev, int minibatchSize, double rate) {
		double[][] dZ = new double[minibatchSize][nOut];
		double[][] grad_W = new double[nOut][nIn];
		double[]   grad_b = new double[nOut];
		
		//采用SGD训练
		//通过反向传播误差得到权重和偏差的梯度
		for(int n = 0; n < minibatchSize; n++) {
			for(int i = 0; i < nOut; i++) {
				for(int j = 0; j < dY[0].length; j++) {
					dZ[n][i] += Wprev[j][i] * dY[n][j];
				}
				dZ[n][i] *= dactivation.apply(Z[n][i]);
				
				for(int k = 0; k < nIn; k++) {
					grad_W[i][k] += dZ[n][i] * X[n][k];
				}
				grad_b[i] += dZ[n][i];
			}
		}
		
		//更新梯度
		for(int i = 0; i < nOut; i++) {
			for(int j = 0; j < nIn; j++) {
				weight[i][j] -= rate * grad_W[i][j] / minibatchSize;
			}
			bias[i] -= rate * grad_b[i] / minibatchSize;
		}
		
		return dZ;
	}
}
