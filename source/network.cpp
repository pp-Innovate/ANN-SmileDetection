#include "network.h"

CascadeClassifier *faceCascade;  //人脸检测级联分类器
Ptr<ANN_MLP> neuralNetwork;      //神经网络

//初始化一个用于训练的神经网络
//参数：无
//返回：0 - 正常，-1 - 异常
int networkInit(void)
{
	/*加载人脸检测分类器*/
	faceCascade = new CascadeClassifier();
	if(!faceCascade->load("./haarcascade_frontalface_alt.xml"))
	{
		printf("Can't load face cascade.\n");
		return -1;
	}

	/*设置神经网络参数*/
    neuralNetwork = ANN_MLP::create();
    Mat layerSize = (Mat_<int>(1,3)<<2500,50,2);                                                      //网络层数及神经元数
    neuralNetwork->setLayerSizes(layerSize);
    neuralNetwork->setActivationFunction(ANN_MLP::SIGMOID_SYM,1.0,1.0);                               //激活函数
    neuralNetwork->setTrainMethod(ANN_MLP::BACKPROP,0.0001,0.001);                                    //训练方法
    neuralNetwork->setTermCriteria(TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30000,0.0001)); //迭代终止条件

	return 0;
}

//初始化一个用于分类的神经网络
//参数：神经网络数据文件名
//返回：0 - 正常，-1 - 异常
int networkInit(const char *fileName)
{
	/*加载人脸检测分类器*/
    faceCascade = new CascadeClassifier();
    if(!faceCascade->load("./haarcascade_frontalface_alt.xml"))
    {
        printf("Can't load face cascade.\n");
        return -1;
    }

	/*从指定文件中加载神经网络数据*/
    printf("Loading network form %s\n",fileName);
    neuralNetwork = Algorithm::load<ANN_MLP>(fileName);

    return 0;
}

//训练神经网络并保存到指定的文件
//参数：正样本容器，负样本容器，保存神经网络数据的文件名
//返回：0 - 正常，-1 - 异常
int networkTrain(std::vector<Mat> *posFaceVector,std::vector<Mat> *negFaceVector,const char *fileName)
{
	/*初始化训练数据矩阵和类标矩阵*/
	//训练数据矩阵每行为一个样本，行数=正样本数+负样本数，列数=样本维数=样本宽度x样本高度=输入层神经元个数
	//类标矩阵每行为一个样本的类标，行数同上，列数=2=输出层神经元个数，(1,0)代表正样本，(0,1)代表负样本
    Mat trainDataMat(posFaceVector->size()+negFaceVector->size(),2500,CV_32FC1);
    Mat labelMat = Mat::zeros(posFaceVector->size()+negFaceVector->size(),2,CV_32FC1);

	/*生成训练数据矩阵和类标矩阵*/
    for(unsigned int i = 0;i < posFaceVector->size();i++)
    {
        posFaceVector->at(i).reshape(0,1).convertTo(trainDataMat.row(i),CV_32FC1,0.003921569,0);  //0.003921569=1/255
        labelMat.at<float>(i,0) = 1;
    }
    for(unsigned int i = 0;i < negFaceVector->size();i++)
    {
        negFaceVector->at(i).reshape(0,1).convertTo(trainDataMat.row(i+posFaceVector->size()),CV_32FC1,0.003921569,0);
        labelMat.at<float>(i+posFaceVector->size(),1) = 1;
    }

	/*训练网络*/
    neuralNetwork->train(trainDataMat,ROW_SAMPLE,labelMat);

	/*保存网络数据*/
    neuralNetwork->save(fileName);

	return 0;
}

//用训练好的神经网络进行分类
//参数：待分类的样本
//返回：1 - 正样本（笑脸），0 - 负样本（非笑脸），-1 - 未检测到人脸
int networkClassify(Mat *image)
{
	/*人脸检测*/
    Mat gray,roi,classifyDataMat,classifyResultMat;

    std::vector<Rect> faces;

    cvtColor(*image,gray,CV_BGR2GRAY);   //将样本转换为灰度图
    //equalizeHist(gray,gray);             //直方图均衡化

    faceCascade->detectMultiScale(
            gray,
            faces,
            1.1,
            4,
            CASCADE_SCALE_IMAGE|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH,
            Size(20,20));                //在样本中检测人脸，只检测最大的人脸

    if(faces.size() < 1)                 //样本中检测不到人脸
        return -1;

    roi = gray(faces[0]);
    resize(roi,roi,Size(50,50),0,0,CV_INTER_LINEAR);  //将检测到的人脸区域缩放到50x50像素

	/*对检测到的人脸进行分类，即判断是笑脸还是非笑脸*/
	roi = Mat_<float>(roi);
    roi.reshape(0,1).convertTo(classifyDataMat,CV_32FC1,0.003921569,0);  //转换为待分类数据
    neuralNetwork->predict(classifyDataMat,classifyResultMat);           //输入到神金网络进行分类
    cout<<classifyResultMat<<endl;                                       //打印分类结果矩阵（二位行向量）

    if(classifyResultMat.at<float>(0,0) > classifyResultMat.at<float>(0,1))
        return 1;                         //若分类结果矩阵的第一个元素大于第二个元素则认为是正样本（笑脸）
    else
        return 0;                         //否则为负样本（非笑脸）
}

//从指定图像中检测人脸
//参数：图像文件名，存放人脸区域的矩阵
//返回：1 - 成功检测到人脸，0 - 未检测到人脸，-1 - 异常
int detectFaceFormImg(const char *imgFilename,Mat *faceRoi)
{
    Mat img,gray,roi;
	
    std::vector<Rect> faces;

	/*读取图像文件*/
	img = imread(imgFilename);
	if(img.empty())
	{
		printf("Can't read file %s\n",imgFilename);
		return -1;
	}

	cvtColor(img,gray,CV_BGR2GRAY);  //将图像转换为灰度图
	//equalizeHist(gray,gray);

	faceCascade->detectMultiScale(
			gray,
            faces,
			1.1,
			6,
			CASCADE_SCALE_IMAGE|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH,
			Size(20,20));            //在图像中检测人脸

    if(faces.size() < 1)             //未检测到人脸
	{
		printf("Can't find face in %s, skip it.\n",imgFilename); 
		return 0;
	}

    roi = gray(faces[0]);                                //人脸区域ROI
    resize(roi,roi,Size(50,50),0,0,CV_INTER_LINEAR);     //将人脸区域ROI缩放到50x50像素

    *faceRoi = roi.clone();                              //将人脸区域ROI拷贝到输出矩阵

	return 1;
}
