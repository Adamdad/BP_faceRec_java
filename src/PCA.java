import java.lang.reflect.Array;
import java.util.Arrays;

import javax.sound.sampled.Mixer;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

class PCA{

	public void testJama(){
		double[] cloumnwise = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
		Matrix aMatrix=new Matrix(cloumnwise,4);//构造矩阵
		aMatrix.print(aMatrix.getColumnDimension(),aMatrix.getRowDimension());
		PCA_resolve(aMatrix, 2);
	}
	
	public Matrix PCA_resolve(Matrix image,int K) {//一列代表一个图片，一行代表一个字段
		
		image=MinusAverage(image);
		Matrix Var = image.times(image.transpose());
		Var=Var.times((double)1/image.getColumnDimension());
		EigenvalueDecomposition Eig = Var.eig();
		Matrix D = Eig.getD();
		D.print(D.getColumnDimension(), D.getRowDimension());
		int[] indexes = this.getIndexesOfKEigenvalues(D,K);
		for (int i = 0; i < indexes.length; i++) {
			System.out.println(indexes[i]);
		}
		Matrix V = Eig.getV();
		Matrix selectedEignVectors = V.getMatrix(0, V.getRowDimension()-1,indexes);
		selectedEignVectors.print(selectedEignVectors.getColumnDimension(), selectedEignVectors.getRowDimension());
		return selectedEignVectors;
	}
	
	// 获得特征值最大的K个特征值
	private class mix implements Comparable {
		int index;
		double value;

		mix(int i, double v) {
			index = i;
			value = v;
		}

		public int compareTo(Object o) {
			double target = ((mix) o).value;
			if (value > target)
				return -1;
			else if (value < target)
				return 1;

			return 0;
		}
	}
	private int[] getIndexesOfKEigenvalues(Matrix d, int k) {
		// TODO Auto-generated method stub
		mix[] mixes = new mix[d.getColumnDimension()];
		for (int i = 0;i<d.getColumnDimension();i++)
		{
			mixes[i] = new mix(i, d.get(i, i));
		}
		
		Arrays.sort(mixes);
		
		int[] result =new int[k];
		for (int j = 0; j < k; j++) {
			result[j]=mixes[j].index;
		}
		return result;
	}

	public Matrix MinusAverage(Matrix image) {
		double temp=0;
		double[] average=new double[image.getRowDimension()];
		for(int i = 0;i<image.getRowDimension();i++)//每一行
		{	
			for(int j=0;j<image.getColumnDimension();j++)//每一列	
			{	
				temp+=(double)image.get(i, j);
			}
			average[i]=temp/image.getColumnDimension();
		}
		for(int i = 0;i<image.getRowDimension();i++)//每一行
		{	
			for(int j=0;j<image.getColumnDimension();j++)//每一列	
			{	
				image.set(i, j,image.get(i, j)-average[i]);
			}
		}
		return image;
	}
}