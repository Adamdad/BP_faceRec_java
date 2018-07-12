import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Writer;
import java.nio.file.attribute.UserPrincipal;
import java.util.Collection;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.*;

import Jama.Matrix;

public class FaceRecognition {
	private int modeltype = 0;
	private int all = 624;										// 总的图片数目
	private int trainSize = 500;								// 训练集大小
	private int picSize = 128*120;								// 图像大小
	private int inputSize = picSize;							// 输入层大小
	private int outputSize = 2; 								// 输出层大小
	private int hiddenSize = 10;								// 隐藏层2节点个数
	private byte[] imageinfor = new byte[15373];				// 存放图像信息
	
	private double[][] v_inputWeight =new double[hiddenSize][picSize + 1];
	private double[][] v_hiddenWeight =new double[outputSize][hiddenSize + 1];
	
	private double[][] alpha1 = new double[outputSize][hiddenSize+1];			// 隐藏层2调整的梯度
	private double[] alpha2 = new double[outputSize];										// 输出层调整的梯度

	private double[][] inputWeight = new double[hiddenSize][picSize + 1];// 输入层->隐藏层1参数
	private double[][] hiddenWeight = new double[outputSize][hiddenSize + 1];// 隐藏层2->输出的参数
	
	private double[] input= new double[inputSize+1];			// 归一化后的图像信息
	private double[] hiddenOutput = new double[hiddenSize + 1]; // 隐藏层输出
	private double[] output = new double[outputSize];			// 输出层							/
		
	private double ci = 0.01;									// 学习率
	private double mo = 0.3;
	private double[] opt = new double[outputSize];				// 期望输出
	Random random = new Random();
	private double [] pro;
	public void savemodal() throws IOException
	{
		File file = new File("J://Learning//人工智能//model//"+ "model"+modeltype+".txt");
		if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
		Writer out = new FileWriter(file);
		
		// 保存inputWeight
	    for (int i = 0; i < hiddenSize; i++)
	    {    
	        for (int j = 0; j < inputSize+1; j++) 
	        {
	        	out.write(inputWeight[i][j]+"\n");
	        }
	    }
	    
	    //保存hiddenWeight
	    for (int i = 0; i < outputSize; i++) {
	    	for (int j = 0; j < hiddenSize + 1; j++) {
	    		out.write(hiddenWeight[i][j]+"\n");
		    }
		}

        out.close();
	}
	
	public FaceRecognition() throws IOException {
		train_expression();  
	}
	public static void main(String[] args) throws IOException{
		System.out.println("BP Face-Recognition！");
		FaceRecognition faceRecognition=new FaceRecognition();
	}
	// 初始化
	public void init() {
		//初始化各参数权值
		
		//初始化输入层->隐藏层参数
	    for (int i = 0; i < hiddenSize; i++) {
//	    	double temp = 0.0;
	        for (int j = 0; j < inputSize + 1; j++){
	            inputWeight[i][j] = random.nextDouble() * 2 - 1;//random.nextGaussian();
//	            temp += inputWeight[i][j];
	        }
//	        for (int j = 0; j < inputSize + 1; j++){
//	            inputWeight[i][j] /= temp;
//	        }
	    }
	    
	    //初始化隐藏层->输出的参数
	    for (int i = 0; i < outputSize; i++) {
//	    	double temp = 0.0;
	    	for (int j = 0; j < hiddenSize + 1; j++) {
		        hiddenWeight[i][j] = random.nextDouble() * 2 - 1;//random.nextGaussian();
//		        temp += hiddenWeight[i][j];
		    }
//    	`	for (int j = 0; j < hiddenSize + 1; j++) {
//        		hiddenWeight[i][j] /= temp;
//  		}
		}
	    
	}

	// sigmoid
	private double Sigmoid(double x) {
	    return 1.0d / (1.0d + Math.exp(-x));
	}


	// 图像文件读入
	public void PGMReader(String filename) {
		
			File file = new File(filename);
			 try {
			        RandomAccessFile in = new RandomAccessFile(file, "r");
			        in.read(imageinfor);
			        in.close();
			    } catch (Exception e) {
			        e.printStackTrace();
			    }
			 for (int j= 0; j < picSize; j++) {
			        int temp = (int) imageinfor[j + 13];
			        input[j]=(double) temp / 255;
			    }
			 input[inputSize] = 1.0;
		}    
	

	public void PGMReader(File file) {
		
			File file2 = file;
			try {
		        RandomAccessFile in = new RandomAccessFile(file, "r");
		        in.read(imageinfor);
		        in.close();
		    } catch (Exception e) {
		        e.printStackTrace();
		    }
		    for (int i = 0; i < picSize; i++) {
		    	int temp = (int) imageinfor[i + 13];
		    	//System.out.println(i);
		        input[i]=(double) temp / 255;
		    }
		    input[inputSize] = 1.0;
		}
	    

	//设置预期输出
	public void setOpt(double[] opt) {
	    this.opt = opt;
	}

	private double Distance(double [] opt) {
		double temp = 0;
		double l = 0;
		for (int i = 0; i < outputSize; i++) {
			l += Math.pow(output[i], 2);
			temp += Math.pow(output[i]-opt[i],2);
		}
		temp /= l;
		return temp;
	}
	//前向传播过程
	private void forward() {
		//向前计算output
		
		//输入 到 隐藏层
	    for (int i = 0; i < hiddenSize; i++) {
	        double temp = 0;
	        for (int j = 0; j < inputSize + 1; j++) {
	            temp += input[j] * inputWeight[i][j];
	        }
	        hiddenOutput[i] = Sigmoid(temp);
	    }
	    hiddenOutput[hiddenSize] = 1.0;
	    
	 

	    //隐藏层 到 输出
	    for (int i = 0; i < outputSize; i++) {
	    	double temp = 0;
	    	for (int j = 0; j < hiddenSize + 1; j++) {
	    		 temp += hiddenOutput[j] * hiddenWeight[i][j];
		    }
	    	output[i] = Sigmoid(temp);
		}
	}

	//反向传播过程
	public void BP() {
	    
		for (int C = 0; C < outputSize; C++) {
			alpha2[C] = (opt[C] - output[C]) * output[C] * (1 - output[C]);//alpha2=误差对隐藏层权值的偏导数
		    
		    for (int i = 0; i < hiddenSize; i++) {
		    	 alpha1[C][i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * alpha2[C] * hiddenWeight[C][i];
		    }
		}

		for (int C = 0; C < outputSize; C++) {
		    // 反向传播
		    for (int i = 0; i < hiddenSize; i++)
		    {   
		    	//alpha1[C][i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * alpha2[C] * hiddenWeight[C][i];
		    	v_hiddenWeight[C][i] = mo * v_hiddenWeight[C][i]+ hiddenOutput[i] * alpha2[C] * ci;
		        hiddenWeight[C][i] += v_hiddenWeight[C][i];
		        for (int j = 0; j < inputSize + 1; j++) 
		        {
		        	v_inputWeight[i][j] = mo*v_inputWeight[i][j]+input[j] * alpha1[C][i] * ci;
		        	inputWeight[i][j] += v_inputWeight[i][j];
		        }
		    }
		    v_hiddenWeight[C][hiddenSize] = mo * v_hiddenWeight[C][hiddenSize] + hiddenOutput[hiddenSize] * alpha2[C] * ci;
		    hiddenWeight[C][hiddenSize] += v_hiddenWeight[C][hiddenSize];
		}
	    
	}

	 public void Randomsort(File[] faceList){ 
		 File tmp=null;
	        for(int i=0;i<faceList.length;i++){  
	            int p = random.nextInt(i+1);  
	           // System.out.println("i==="+i+"p==="+p);  
	            tmp = faceList[i];  
	            faceList[i] = faceList[p];  
	            faceList[p] = tmp;  
	        }   
	    }  
	 
	public void train_pose() throws IOException {
		  
		    String facePath = "J://Learning//人工智能//128_120";
		    File faceFile = new File(facePath);
		    File[] faceList = faceFile.listFiles();
		    Randomsort(faceList);
		    File[] train=new File[trainSize];
		    File[] test=new File[all-trainSize];
		    for (int j = 0; j < all ; j++) 
	        {
		    	if (j<trainSize) {
		    		train[j] = faceList[j];
				} else {
					test[j-trainSize] = faceList[j-trainSize];
				}
	        }
		    init();
		    int choose;
		    String[] Names = {"straight", "left", "right", "up"};
		    double[][] output = {
		    		{0.0,0.0},
		    		{0.0,1.0},
		    		{1.0,0.0},
		    		{1.0,1.0}
		    		};
		    pro =new double [3000];//保存每一次的正确率
		    for(int i =0;i<3000;i++){
		        int right = 0;
		        String type;
		        for (int j = 0; j < trainSize; j++) 
		        {
		        	type=train[j].getName().split("\\.")[0].split("\\_")[1];
		        	this.PGMReader(train[j]);
		        	this.forward();
		        	double tempDis = Distance(output[0]);
		        	choose = 0;
		        	for (int j2 = 1; j2 < output.length; j2++) {
		        		if (Distance(output[j2])<tempDis) 
		        		{
							choose = j2;
							tempDis = Distance(output[j2]);
						} 
					}
		        	
		        	for (int k = 0; k < output.length; k++) {
		        		if(type.equals(Names[k]))
		        		{ 
			                this.setOpt(output[k]);	                
			                this.BP();
			                if (choose==k)
			                {
			                	right++;						//正好第K类，正确  
			                }
			                break;
			            }
					}
		        }

		        pro[i] = (double) right / trainSize;
		        System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
		        if(i%100==0)
		        {
		        	modeltype = i;
		        	savemodal();
		        	System.out.println("保存模型");
		        }
		        
		        if(pro[i]>=0.95){
		            System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
		            modeltype = i;
		            savemodal();
		        	System.out.println("保存最终模型");
		            break;
		        }        
		    }
		    
		    double testpro;
	        int right = 0;
	        String type;

	        for (int j = 0; j <(all - trainSize); j++) {
	        	type=test[j].getName().split("\\.")[0].split("\\_")[1];
	            { // 正例测试
	                this.PGMReader(test[j]);
	                this.forward();
	                double tempDis = Distance(output[0]);
	                choose = 0;
		        	for (int j2 = 1; j2 < output.length; j2++) { //网络输出是第choose类
		        		if (Distance(output[j2])<tempDis) 		
		        		{
							choose = j2;
							tempDis = Distance(output[j2]);
						} 
					}
		        	
		        	for (int k = 0; k < output.length; k++) {
		        		if(type.equals(Names[k])&&choose==k){ // open
		        			right++;						//正好第K类，正确
			                break;
			            }
					}
	            }
	        }
	        testpro = (double) right / (all - trainSize);
	        System.out.println("测试集的估算正确率为：" + testpro);
		   
		}
	
	public void train_people() throws IOException {

		    String facePath = "J://Learning//人工智能//128_120";
		    File faceFile = new File(facePath);
		    File[] faceList = faceFile.listFiles();
		    Randomsort(faceList);
		    File[] train=new File[trainSize];
		    File[] test=new File[all-trainSize];
		    for (int j = 0; j < all ; j++) 
	        {
		    	if (j<trainSize) {
		    		train[j] = faceList[j];
				} else {
					test[j-trainSize] = faceList[j-trainSize];
				}
	        }
		    init();
		    int choose = 0;
		    String[] Names = {"an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman",
		    		"karyadi", "kawamura", "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi",
		    		"sz24", "tammo"};
		    double[][] output = {
		    		{0.0,0.0,0.0,0.0,0.0},
		    		{0.0,0.0,0.0,0.0,1.0},
		    		{0.0,0.0,0.0,1.0,0.0},
		    		{0.0,0.0,0.0,1.0,1.0},
		    		{0.0,0.0,1.0,0.0,0.0},
		    		{0.0,0.0,1.0,0.0,1.0},
		    		{0.0,0.0,1.0,1.0,0.0},
		    		{0.0,0.0,1.0,1.0,1.0},
		    		{0.0,1.0,0.0,0.0,0.0},
		    		{0.0,1.0,0.0,0.0,1.0},
		    		{0.0,1.0,0.0,1.0,0.0},
		    		{0.0,1.0,0.0,1.0,1.0},
		    		{0.0,1.0,1.0,0.0,0.0},
		    		{0.0,1.0,1.0,0.0,1.0},
		    		{0.0,1.0,1.0,1.0,0.0},
		    		{0.0,1.0,1.0,1.0,1.0},
		    		{1.0,0.0,0.0,0.0,0.0},
		    		{1.0,0.0,0.0,0.0,1.0},
		    		{1.0,0.0,0.0,1.0,0.0},
		    		{1.0,0.0,0.0,1.0,1.0},
		    		};
		    pro =new double [3000];//迭代151次每一次的正确率
		    for(int i =0;i<3000;i++){
		        int right = 0;
		        String type;
		        for (int j = 0; j < trainSize; j++) 
		        {
		        	type=train[j].getName().split("\\.")[0].split("\\_")[0];
		        	this.PGMReader(train[j]);
		        	this.forward();
		        	double tempDis = Distance(output[0]);
		        	choose = 0;
		        	for (int j2 = 1; j2 < output.length; j2++) {
		        		if (Distance(output[j2])<tempDis) 
		        		{
							choose = j2;
							tempDis = Distance(output[j2]);
						} 
					}
		        	
		        	for (int k = 0; k < output.length; k++) {
		        		if(type.equals(Names[k]))
		        		{ 
			                this.setOpt(output[k]);	                
			                this.BP();
			                if (choose==k)
			                {
			                	right++;						//正好第K类，正确
			                }
			                break;
			            }
					}
		        }

		        pro[i] = (double) right / trainSize;
		        System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
		        if(i%100==0)
		        {
		        	modeltype = i;
		        	savemodal();
		        	System.out.println("保存模型");
		        }
		        
		        if(pro[i]>=0.95){
		            System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
		            modeltype = i;
		            savemodal();
		        	System.out.println("保存最终模型");
		            break;
		        }        
		    }
		    
		    double testpro;
	        int right = 0;
	        String type;

	        for (int j = 0; j <(all - trainSize); j++) {
	        	type=test[j].getName().split("\\.")[0].split("\\_")[0];
	            { // 正例测试
	                this.PGMReader(test[j]);
	                this.forward();
	                double tempDis = Distance(output[0]);
	                choose = 0;
		        	for (int j2 = 1; j2 < output.length; j2++) { //网络输出是第choose类
		        		if (Distance(output[j2])<tempDis) 		
		        		{
							choose = j2;
							tempDis = Distance(output[j2]);
						} 
					}
		        	
		        	for (int k = 0; k < output.length; k++) {
		        		if(type.equals(Names[k])&&choose==k){ // open
			                right++;						//正好第K类，正确
			                break;
			            }
					}
	            }
	        }
	        testpro = (double) right / (all - trainSize);
	        System.out.println("测试集的估算正确率为：" + testpro);
		   
		}
	
	public void train_eyes() throws IOException {

	    String facePath = "J://Learning//人工智能//128_120";
	    File faceFile = new File(facePath);
	    File[] faceList = faceFile.listFiles();
	    Randomsort(faceList);
	    File[] train=new File[trainSize];
	    File[] test=new File[all-trainSize];
	    for (int j = 0; j < all ; j++) 
        {
	    	if (j<trainSize) {
	    		train[j] = faceList[j];
			} else {
				test[j-trainSize] = faceList[j-trainSize];
			}
	    	
        }
	    init();
	    int choose;
	    String[] Names = {"open", "sunglasses"};
	    double[][] output = {
	    		{1.0,0.0},
	    		{0.0,1.0}
	    		};
	    pro =new double [2000];//最多迭代2000次
	    for(int i =0;i<2000;i++){
	        int right = 0;
	        String type;
	        for (int j = 0; j < trainSize; j++) 
	        {
	        	choose = 0;
	        	type=train[j].getName().split("\\.")[0].split("\\_")[3];
	        	this.PGMReader(train[j]);
	        	this.forward();
	        	
	        	double tempDis = Distance(output[0]);
	        	for (int j2 = 1; j2 < outputSize; j2++) {		//选择一个距离最小的向量编号为结果
	        		if (Distance(output[j2])<tempDis) 
	        		{
						choose = j2;
						tempDis = Distance(output[j2]);
					} 
				}
	        	
	        	for (int k = 0; k < outputSize; k++) {
	        		if(type.equals(Names[k])){ // open
		                this.setOpt(output[k]);	                
		                this.BP();
		                if (choose==k)
		                {
		                	right++;						//正好第K类，正确
		                }
		                break;
		            }
				}
	        }

	        pro[i] = (double) right / trainSize;
	        System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
	        if(i%20==0)
	        {
	        	modeltype = i;
	        	//savemodal();
	        	System.out.println("保存模型");
	        }
	        
	        if(pro[i]>=0.95){
	            System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
	            modeltype = i;
	            savemodal();
	        	System.out.println("保存最终模型");
	            break;
	        }        
	    }
	    
	    double testpro;
        int right = 0;
        String type;

        for (int j = 500; j <all; j++) {
        	type=test[j-trainSize].getName().split("\\.")[0].split("\\_")[3];
            { // 正例测试
            	choose = 0;
                this.PGMReader(test[j-trainSize]);
                this.forward();
                
                double tempDis = Distance(output[0]);
	        	for (int j2 = 1; j2 < outputSize; j2++) {		//选择一个距离最小的向量编号为结果
	        		if (Distance(output[j2])<tempDis) 
	        		{
						choose = j2;
						tempDis = Distance(output[j2]);
					} 
				}
	        	
                if (type.equals("open")&&choose == 0)
                    right++;
                else if (type.equals("sunglasses")&&choose == 1)
                    right++;
            }
        }
        testpro = (double) right / (all - trainSize);
        System.out.println("测试集的估算正确率为：" + testpro);
	}
	
	public void train_expression() throws IOException {
		  
	    String facePath = "J://Learning//人工智能//128_120";
	    File faceFile = new File(facePath);
	    File[] faceList = faceFile.listFiles();
	    Randomsort(faceList);
	    File[] train=new File[trainSize];
	    File[] test=new File[all-trainSize];
	    for (int j = 0; j < all ; j++) 
        {
	    	if (j<trainSize) {
	    		train[j] = faceList[j];
			} else {
				test[j-trainSize] = faceList[j-trainSize];
			}
        }
	    init();
	    int choose;
	    String[] Names = {"neutral", "happy", "sad", "angry"};
	    double[][] output = {
	    		{0.0,0.0},
	    		{0.0,1.0},
	    		{1.0,0.0},
	    		{1.0,1.0}
	    		};
	    pro =new double [3000];//保存每一次的正确率
	    for(int i =0;i<3000;i++){
	        int right = 0;
	        String type;
	        for (int j = 0; j < trainSize; j++) 
	        {
	        	type=train[j].getName().split("\\.")[0].split("\\_")[2];
	        	this.PGMReader(train[j]);
	        	this.forward();
	        	double tempDis = Distance(output[0]);
	        	choose = 0;
	        	for (int j2 = 1; j2 < output.length; j2++) {
	        		if (Distance(output[j2])<tempDis) 
	        		{
						choose = j2;
						tempDis = Distance(output[j2]);
					} 
				}
	        	
	        	for (int k = 0; k < output.length; k++) {
	        		if(type.equals(Names[k]))
	        		{ 
		                this.setOpt(output[k]);	                
		                this.BP();
		                if (choose==k)
		                {
		                	right++;						//正好第K类，正确  
		                }
		                break;
		            }
				}
	        }

	        pro[i] = (double) right / trainSize;
	        System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
	        if(i%100==0)
	        {
	        	modeltype = i;
	        	savemodal();
	        	System.out.println("保存模型");
	        }
	        
	        if(pro[i]>=0.95){
	            System.out.println("第"+i+"次迭代估算正确率为：" + pro[i]);
	            modeltype = i;
	            savemodal();
	        	System.out.println("保存最终模型");
	            break;
	        }        
	    }
	    
	    double testpro;
        int right = 0;
        String type;

        for (int j = 0; j <(all - trainSize); j++) {
        	type=test[j].getName().split("\\.")[0].split("\\_")[2];
            { // 正例测试
                this.PGMReader(test[j]);
                this.forward();
                double tempDis = Distance(output[0]);
                choose = 0;
	        	for (int j2 = 1; j2 < output.length; j2++) { //网络输出是第choose类
	        		if (Distance(output[j2])<tempDis) 		
	        		{
						choose = j2;
						tempDis = Distance(output[j2]);
					} 
				}
	        	
	        	for (int k = 0; k < output.length; k++) {
	        		if(type.equals(Names[k])&&choose==k){ // open
	        			right++;						//正好第K类，正确
		                break;
		            }
				}
            }
        }
        testpro = (double) right / (all - trainSize);
        System.out.println("测试集的估算正确率为：" + testpro);
	   
	}

	public void inmodel() throws IOException {
		File fmodel=new File("J://Learning//人工智能//model//model72.txt");
		FileReader fin=new FileReader(fmodel);
		String str  = null;
		BufferedReader br=new BufferedReader(fin);
		
		//读入inputWeight
	    for (int i = 0; i < hiddenSize; i++)
	    {    
	        for (int j = 0; j <=picSize; j++) 
	        {
	        	str = br.readLine(); 
	        	inputWeight[i][j]=Double.parseDouble(str);
	        }
	    }
	   
	    //读入hiddenWeight
	    for (int i = 0; i < outputSize; i++) {
	    	for (int j = 0; j < hiddenSize + 1; j++) {
	    		hiddenWeight[i][j]=Double.parseDouble(str);
		    }
		}

        br.close();
        fin.close();
	}
	public void test() throws IOException {
		inmodel();
		String facePath = "J://Learning//人工智能//128_120";
	    File faceFile = new File(facePath);
	    File[] faceList = faceFile.listFiles();
	    init();
	    double testpro;//迭代151次每一次的正确率
	    
	        int right = 0;
	        String type;

	        for (int j = 500; j <624; j++) {
	        	this.PGMReader(faceList[j]);
	        	type=faceList[j].getName().split("\\.")[0].split("\\_")[3];
	            { // 正例测试
	                this.PGMReader(faceList[j]);
	                this.forward();
	                if (type.equals("open")&&output[0] > 0.5)
	                    right++;
	                else if (type.equals("sunglasses")&&output[0] < 0.5)
	                    right++;
	            }
	        }
	        testpro = (double) right / 124;
	        System.out.println("测试集的估算正确率为：" + testpro);
	}
}
