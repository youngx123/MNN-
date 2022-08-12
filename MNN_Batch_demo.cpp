#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/Tensor.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/MNNDefine.h>
#include<MNN/ImageProcess.hpp>
#include <stdio.h>
#include<time.h>
#include<vector>
#include<math.h>

int main()
{
	std::string mnnPaht = "./multi_label_classification_sim.mnn";
	std::string imgFolds = "./test_img";
	std::string saveFolds = "./result";
	int batch_size = 3;


	MNN::Interpreter *model;
	MNN::ScheduleConfig config;
	MNN::Session *session;

	model = MNN::Interpreter::createFromFile(mnnPaht.c_str());
	config.type = MNN_FORWARD_CPU;
	MNN::BackendConfig bnconfig;
	bnconfig.precision = MNN::BackendConfig::Precision_Low;//Precision_Low�� Precision_Normal
	bnconfig.memory = MNN::BackendConfig::Memory_Normal; //Memory_Low
	config.backendConfig = &bnconfig;
	

	std::vector<std::string> fileNames;
	cv::glob(imgFolds,fileNames);

	if (fileNames.size() % 2 == 0)
	{
		printf("pairs images\n");
	}
	else
	{
		printf("odd images", fileNames.size());
		fileNames.pop_back();
		printf("pairs images\n");
	}

	const std::string  COLOR[] = { "white", "red", "blue", "black" };
	const std::string  TYPE[] = { "dress", "jeans", "shirt", "shoe", "bag" };
	
	// get input and output tensor,and resize to new batchsize
	session = model->createSession(config);
	auto input = model->getSessionInput(session, NULL);
	auto shape = input->shape();
	shape[0] = batch_size;

	model->resizeTensor(input, shape);
	model->resizeSession(session);
	MNN::Tensor* output = model->getSessionOutput(session, NULL);


	int image_size = fileNames.size();

	int save_num = 0;
	cv::Mat img1;
	std::string file;
	float use_time = 0;

	// set image mean and std_norm for image processing
	float mean[3] = { 0.f, 0.f, 0.f };
	float std_norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };

	MNN::CV::ImageProcess::Config cfg;
	cfg.filterType = MNN::CV::BILINEAR;
	::memcpy(cfg.mean, mean, sizeof(mean));
	::memcpy(cfg.normal, std_norm, sizeof(std_norm));
	cfg.sourceFormat = MNN::CV::RGB;
	cfg.destFormat = MNN::CV::RGB;

	// inference
	while (fileNames.size())
	{
		std::shared_ptr<MNN::Tensor> inputUser(new MNN::Tensor(input, MNN::Tensor::TENSORFLOW));
		int bpp = inputUser->channel();
		int size_h = inputUser->height();
		int size_w = inputUser->width();


		clock_t start_time, end_time;
		std::vector<cv::Mat> img_vet;

		clock_t start_read_time, end_read_time;
		start_time = clock();
		for (int batch = 0; batch < batch_size; batch++)
		{
			auto iter = fileNames.begin();
			file = fileNames.at(0);
			fileNames.erase(iter);

			std::cout <<"image file :"<< file << std::endl;
			start_read_time = clock();
			img1 = cv::imread(file);
			end_read_time = clock();

			cv::Mat temp = img1.clone();
			img_vet.push_back(temp);

			int width, height;
			width = img1.cols;
			height = img1.rows;

			MNN::CV::Matrix trans;
			trans.setScale((float)(width - 1) / (size_w - 1), (float)(height - 1) / (size_h - 1));

			std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(cfg));
			pretreat->setMatrix(trans);
			/*			
			std::cout << "image step value  " << width << "  " << height << std::endl;
			std::cout <<"image step value "<< img1.step[1] <<"  "<< img1.step[0] << std::endl;
			*/
			pretreat->convert(img1.data, width, height, img1.step[0],
				inputUser->host<float>() + inputUser->stride(0) * batch,
				size_w, size_h, bpp, img1.step[0], inputUser->getType());
		}
		input->copyFromHostTensor(inputUser.get());
		model->runSession(session);
		auto dimType = output->getDimensionType(); //TENSORFLOW, CAFFE

		std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
		output->copyToHostTensor(outputUser.get());
		end_time = clock();
		float read_time = (end_read_time - start_read_time) * 1000 / CLOCKS_PER_SEC;
		float total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;

		use_time = use_time + total_time - read_time * batch_size;
		std::cout << "batch size = " << batch_size << "  inferece time :" 
			      << total_time - read_time * batch_size << std::endl;
		std::cout << "input shape : " <<
			input->shape()[0] << "   " <<
			input->shape()[1] << "   " <<
			input->shape()[2] << "   " <<
			input->shape()[3] << std::endl;

		std::cout << "result shape : " <<
			outputUser->shape()[0] << "   " <<
			outputUser->shape()[1] << "\n";

		//std::cout << outputUser->stride(0) << std::endl;
		// post-porcessing and save the result
		auto type = outputUser->getType();
		int num_lenth = output->stride(0);
		cv::Mat saveimag;
		for (int batch = 0; batch < batch_size; batch++)
		{		
			float max_color = 0;
			float max_type = 0;
			int color_index = 0;
			int type_index=0;
			auto values = outputUser->host<float>() + batch * outputUser->stride(0);
			for (int k = 0; k < num_lenth; k++)
			{
				float temp = values[k];
		    		if (k < 4)
				{
					if (max_color < temp)
					{
						max_color = temp;
						color_index = k;
					}
				}
				else
				{
					if (max_type < temp)
					{
						max_type = temp;
						type_index = k-4;
					}
				}
			}
			std::cout<< std::endl;
			saveimag = img_vet.at(batch);
			std::string text = COLOR[color_index] + " "+ std::to_string(max_color);
			std::string text2 = TYPE[type_index] + " " + std::to_string(max_type);
			cv::putText(saveimag, text, cv::Point(200, 160), cv::FONT_HERSHEY_SIMPLEX,  3, cv::Scalar(0,255,0), 2);
			cv::putText(saveimag, text2, cv::Point(200, 260), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
			save_num = save_num + 1;
			std::string savefile = saveFolds + "/" + std::to_string(save_num) +".png";
			//std::cout << "save file " << savefile << std::endl;
			cv::imwrite(savefile, saveimag);
		}
	
		
	}

	std::cout << "total use time  : " << use_time << std::endl;
	std::cout << image_size << " images, batch size is  " << batch_size << 
		       "  use mean time " << use_time / image_size << std::endl;
	return 0;
}