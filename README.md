# ANN-SmileDetection

这是一个使用BP神经网络进行笑脸检测的程序
作者：PP(1631552) 1631552pp@tongji.edu.cn ,YZH(1631546), LW(1632407)

工程结构：
	source -- 源代码和Makefile
	trainset -- 训练集
	networks -- 神经网络数据文件

功能：
	训练 -- 使用训练样本路径下的所有图片训练一个BP神经网络，并把训练好的神经网络保存到文件
	        默认训练样本路径分别为trainset/positive和trainset/negative
			默认神经网络数据文件为networks/default.xml
	预览 -- 预览摄像头拍摄的实时图像，不对其做任何处理
	        图像分辨率为640x480
			默认摄像头设备为/dev/video0
	检测 -- 从文件中加载神经网络分类器，并在摄像头拍摄的实时图像中进行笑脸检测，若检测到笑脸则在图像周围绘制一个红色矩形
			图像分辨率为320x240
			默认神经网络数据文件为networks/default.xml
			默认摄像头设备为/dev/video0

依赖：
	libv4l-dev
	opencv 3.0

编译：
	cd source
	make clean
	make 

运行：
	训练 -- ./smileFaceDetection -t                                                           //使用默认训练样本路径和神经网络数据文件名
	        ./smileFaceDetection -t -T Positive-Samples-Location -F Negtive-Samples-Location  //指定训练样本路径
			./smileFaceDetection -t -N Network-File-Name                                      //指定神经网络数据文件名
	预览 -- ./smileFaceDetection -p                                                           //使用默认摄像头设备
	        ./smileFaceDetection -p -d Device-Name                                            //指定摄像头设备
	检测 -- ./smileFaceDetection -e                                                           //使用默认摄像头设备和神经网络数据文件
	        ./smileFaceDetection -e -d Device-Name                                            //指定摄像头设备
			./smileFaceDetection -e -N                                                        //指定神经网络数据文件
