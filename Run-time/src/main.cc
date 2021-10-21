/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <unistd.h>
#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

#define ANCHS 9
#define GRID_SIZE 7
#define CLASSES 1
#define BOX_PRED (5+CLASSES)
#define CELL_PRED ((BOX_PRED)*ANCHS)
#define DPU_OUTPUT (GRID_SIZE*GRID_SIZE*CELL_PRED)

#define THRESHOLD_IOU 0.5
#define THRESHOLD_SCORE 0.6
#define THRESHOLD_BOXES 100
#define OFFSET_MS 1000000
#define FRAMERATE 24
#define DELAY_NEWFRAME (OFFSET_MS/FRAMERATE)
#define CAPTURE_TIME 10 //seconds
#define TOTAL_FRAMES (CAPTURE_TIME*FRAMERATE) //seconds

using namespace std;
using namespace cv;

GraphInfo shapes;

const string baseImagePath = "../images/";
const string wordsPath = "./";

/*
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */

namespace detection
{ 

	float sigmoid(float x)
	{
		return 1/(1+exp(-x));
	}

	float getIOU(float (&xy1)[4], float (&xy2)[4])
	{	
		float x[2];
		float y[2];

		float area1 = (xy1[2]-xy1[0])*(xy1[3]-xy1[1]);
		float area2 = (xy2[2]-xy2[0])*(xy2[3]-xy2[1]);
		
		if(xy1[0] > xy2[0])
			x[0] = xy1[0];
		else
			x[0] = xy2[0];
		
		if(xy1[1] > xy2[1])
			y[0] = xy1[1];
		else
			y[0] = xy2[1];

		if(xy1[2] < xy2[2])
			x[1] = xy1[2];
		else
			x[1] = xy2[2];
		
		if(xy1[3] < xy2[3])
			y[1] = xy1[3];
		else
			y[1] = xy2[1];

		float intersection = (x[1]-x[0])*(y[1]-y[0]);

		return intersection/(area1+area2-intersection);
	}

	class bbox
	{
		private:

			int8_t *tx;
			int8_t *ty;
			int8_t *tw;
			int8_t *th;
			int8_t *ob;
			int8_t *cl[CLASSES];

		public:
		
			float score;
			uint8_t index;
			bool eval = false;
			float xy12[4];
			
			void setBox(int8_t *& predictions, int offset)
			{ 
				tx = &predictions[offset];
				ty = &predictions[offset + 1];
				tw = &predictions[offset + 2];
				th = &predictions[offset + 3];
				ob = &predictions[offset + 4];
				for (uint8_t i = 0; i < CLASSES; i++)
					cl[i] = &predictions[offset + 5 + i];
			}

			void printPredBox()
			{
				std::cout << (int) *tx << (int) *ty << (int) *tw << (int) *th << (int) *ob;
				
				for (uint8_t i = 0; i < CLASSES; i++)
					std::cout << (int) *cl[i];
					std::cout << std::endl;
			}
			

			void printBox()
			{
				std::cout << "x1: " << xy12[0] << ", y1: " << xy12[1] << ", x2: " << xy12[2] << ", y2: " << xy12[3] << ", score: " << score << std::endl;
				
			}			
			float getObjectness(float &scale)
			{
        		//std::cout << "Objectness: " << (float) *ob << " Sigmoidness: " << sigmoid(((float)*ob)*scale) << endl;
				return sigmoid(((float) *ob)*scale);
			}

			void getClassScores(float (&classes)[CLASSES],float &scale)
			{


				for(uint8_t i = 0; i < CLASSES; i++)
					classes[i] = sigmoid(((float) *cl[i])*scale);

        		//std::cout << "Classness: " << (float) *cl[0] << " Sigmoidness: " << sigmoid(((float)*cl[0])*scale) << endl;
			}

			void setXY(uint8_t &gridy, uint8_t &gridx, float (&anchors)[2], float &scale)
			{	
				if(eval == true)
				{
					float xy[2] = {(sigmoid(((float) *tx)*scale)),(sigmoid(((float) *ty)*scale))};
					float wh[2] = {(sigmoid(((float) *tw)*scale)),(sigmoid(((float) *th)*scale))};
					xy[0] = (xy[0] + ((float) gridx))/GRID_SIZE;
					xy[1] = (xy[1] + ((float) gridy))/GRID_SIZE;
					wh[0] = exp(wh[0])*anchors[0];
					wh[1] = exp(wh[1])*anchors[1];
					xy12[0] = xy[0] - wh[0]/2;
					xy12[1] = xy[1] - wh[1]/2;
					xy12[2] = xy[0] + wh[0]/2;
					xy12[3] = xy[1] + wh[1]/2;
				} 
			}

			void setBoxScore(float &scale)
			{
				float objectness = getObjectness(scale);
				if (objectness>THRESHOLD_SCORE)
				{
					
					float classes[CLASSES];
					getClassScores(classes,scale);
					index = 0;
					for (uint8_t i = 0; i < CLASSES; i++)
					{
						if(classes[i]>classes[index])
							index = i;

					}
					float tmpScore = classes[index]*objectness;

					if (tmpScore > THRESHOLD_SCORE)
					{
						eval = true;
						score = tmpScore;

					}
					else
						eval = false;
					
				}
				else
					eval = false;
				
				
			}



	};
	

	bool noIntersection(bbox &Box1, bbox &Box2)
	{
		if ( ( (Box1.xy12[2] < Box2.xy12[0] || Box2.xy12[2] < Box1.xy12[0])) || ( (Box1.xy12[3] < Box2.xy12[1]) || (Box2.xy12[3] < Box1.xy12[1]) ) )
			return false;
		else
			return true;
	}

	void NMS(bbox &Box1, bbox &Box2)
	{
		//std::cout <<(Box1.index == Box2.index) << std::endl;
		if ((Box1.index == Box2.index) && (getIOU(Box1.xy12,Box2.xy12) < THRESHOLD_IOU))
			if (Box1.score > Box2.score)
				Box2.eval = false;
			else
				Box1.eval = false;
	}
	class cell
	{
		private:
		
		public:

			bbox bboxes[ANCHS];
			void setCellBoxes(int8_t *& predictions, int offset)
			{
				for(unsigned int i = 0; i < ANCHS; i++)
					bboxes[i].setBox(predictions, i*BOX_PRED + offset);
			}
			void printPredCell()
			{
				for(unsigned int i = 0; i < ANCHS; i++)
					bboxes[i].printPredBox();
			}
			void setGridScores(float &scale)
			{
				for (uint8_t i = 0; i < ANCHS; i++)
					bboxes[i].setBoxScore(scale);
				
			}

			void setCellXY(uint8_t &gridy, uint8_t &gridx, float (&anchors)[ANCHS][2], float &scale)
			{

				for (uint8_t i = 0; i < ANCHS; i++)
					bboxes[i].setXY(gridy, gridx, anchors[i], scale);
			}
	};

	class grid
	{	
		private:

		public:

			cell cells[GRID_SIZE][GRID_SIZE];


			void grod(int8_t *& predictions)
			{
				for(uint8_t i = 0; i < GRID_SIZE; i++)
					for(uint8_t j = 0; j < GRID_SIZE; j++)
						cells[i][j].setCellBoxes(predictions, i*54*7 + j*54);
			}
			
			void printPredGrid()
			{
				for(uint8_t i = 0; i < GRID_SIZE; i++)
					for(uint8_t j = 0; j < GRID_SIZE; j++)
						cells[i][j].printPredCell();
			}
			void setAllScoresXY(float (&anchors)[ANCHS][2], float &scale)
			{

				for(uint8_t i = 0; i < GRID_SIZE; i++)
					for(uint8_t j = 0; j < GRID_SIZE; j++)
					{
						cells[i][j].setGridScores(scale);
						cells[i][j].setCellXY(i,j, anchors, scale);
					}
			}
			

			
			void doGridNMS(float (&anchors)[ANCHS][2], float &scale)
			{
				
				setAllScoresXY(anchors, scale);
				for (uint8_t i = 0; i < GRID_SIZE; i++)
					for (uint8_t j = 0; j < GRID_SIZE; j++)
						for (uint8_t p = 0; p < ANCHS; p++)
							if (cells[i][j].bboxes[p].eval == true)
								for (uint8_t ii = 0; ii < GRID_SIZE; ii++)
									for (uint8_t jj = 0; jj < GRID_SIZE; jj++)
										for (uint8_t pp = 0; pp < ANCHS; pp++)
											if (cells[ii][jj].bboxes[pp].eval == true)
												if (noIntersection(cells[i][j].bboxes[p], cells[ii][jj].bboxes[pp]) && !(i == ii && j == jj && p==pp))
													NMS(cells[i][j].bboxes[p], cells[ii][jj].bboxes[pp]);			
			}

			void gridPrint(float (&anchors)[ANCHS][2], float &scale)
			{
				doGridNMS(anchors, scale);
				
				for (uint8_t i = 0; i < GRID_SIZE; i++)
					for (uint8_t j = 0; j < GRID_SIZE; j++)
						for (uint8_t p = 0; p < ANCHS; p++)
							if (cells[i][j].bboxes[p].eval == true)
              {
                std::cout << "True box detected at: [" << (int)i << "][" << (int)j << "][" << (int)p << "]" << std::endl;
                cells[i][j].bboxes[p].printBox();
              }
			}
      void drawRectangles(Mat &img, float (&anchors)[ANCHS][2], float &scale)
      {
        doGridNMS(anchors, scale);
		float res = (float)img.cols;
		for (uint8_t i = 0; i < GRID_SIZE; i++)
			for (uint8_t j = 0; j < GRID_SIZE; j++)
				for (uint8_t p = 0; p < ANCHS; p++)
					if (cells[i][j].bboxes[p].eval == true)
					{	

						std::cout << "True box detected at: [" << (int)i << "][" << (int)j << "][" << (int)p << "]" << std::endl;
						cells[i][j].bboxes[p].printBox();
						cv::Point pt1(cells[i][j].bboxes[p].xy12[0]*res, cells[i][j].bboxes[p].xy12[1]*res);
						cv::Point pt2(cells[i][j].bboxes[p].xy12[2]*res, cells[i][j].bboxes[p].xy12[3]*res);
						cv::putText(img, std::to_string(cells[i][j].bboxes[p].score), pt1, cv::FONT_HERSHEY_DUPLEX, 0.5, 
										cv::Scalar(0, 0, 0), 0.7, false);
						cv::rectangle(img, pt1, pt2, cv::Scalar(0, 0, 0));
					}
      }

	};
}
using namespace detection;
class RTS
{
	private:
		std::chrono::_V2::system_clock::time_point start;
		int interval;
	public:
		RTS(int usec)
		{
			interval = usec;
			start = std::chrono::high_resolution_clock::now();
		}
		bool tick()
		{
			if (interval <= std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count())
			{
				start = std::chrono::high_resolution_clock::now();
				return true;
			}else
				return false;
		}

};

void ListImages(string const& path, vector<string>& images) {
  images.clear();
  struct dirent* entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/*
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const& path, vector<string>& kinds) {
  kinds.clear();
  ifstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
}

/*
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const int8_t* data, size_t size, float* result,
                    float scale) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp((float)data[i] * scale);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/*
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float* d, int size, int k, vector<string>& vkinds) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
           vkinds[ki.second].c_str());
    q.pop();
  }
}

/*
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void runResnet50(vart::Runner* runner ) {
  /* Mean value for ResNet50 specified in Caffe prototxt */
  vector<string> kinds, images;

  /* get in/out tensors and dims*/
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);

  /*get shape info*/
  int outSize  = shapes.outTensorList[0].size;
  int inSize   = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth  = shapes.inTensorList[0].width;

  vector<Mat> imageList;
  int batchSize = in_dims[0];

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  int8_t* imageInputs = new int8_t[inSize * batchSize];

  float* softmax = new float[outSize];
  int8_t* FCResult = new int8_t[batchSize * outSize];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  grid grid;
  grid.grod(FCResult);


  float nanks[ANCHS][2] = 
     {{0.02403846, 0.03125   },
      {0.03846154, 0.07211538},
      {0.07932692, 0.05528846},
      {0.07211538, 0.14663461},
      {0.14903846, 0.10817308},
      {0.14182693, 0.28605768},
      {0.27884614, 0.21634616},
      {0.375     , 0.47596154},
      {0.89663464, 0.78365386}};
  /*run with batch*/
    in_dims[0] = 1;
    out_dims[0] = 1;
	VideoCapture cap(-1);
	//int vidHeight = 224;
	//int vidWidth = 224;
	//cap.set(cv::CAP_PROP_FRAME_WIDTH, vidWidth);
	//cap.set(cv::CAP_PROP_FRAME_HEIGHT, vidHeight);
	Mat image;
	VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), FRAMERATE, Size(inHeight, inWidth));
	RTS rts(DELAY_NEWFRAME);
  for (unsigned int n = 0; n < TOTAL_FRAMES; n += 1) {
	
    //Mat image;// = imread("../images/001.jpg");
	cap >> image;
	if (image.empty())
      break;
    /*image pre-process*/
    Mat image2;  //= cv::Mat(inHeight, inWidth, CV_8SC3);
    resize(image, image2, Size(inHeight, inWidth), 0, 0);
    for (int h = 0; h < inHeight; h++) {
      for (int w = 0; w < inWidth; w++) {
        for (int c = 0; c < 3; c++) {
          imageInputs[h * inWidth * 3 + w * 3 + c] =
              (int8_t)((((float)image2.at<Vec3b>(h, w)[c])/255) * input_scale);
        }
      }
    }
    imageList.push_back(image);
    

    /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        FCResult, batchTensors.back().get()));

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    /*run*/
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    
    //grid.gridPrint(nanks, output_scale);
    grid.drawRectangles(image2, nanks, output_scale);
	video.write(image2);

    //imwrite("res.jpg", image);
    imageList.clear();
    inputs.clear();
    outputs.clear();

	while(!rts.tick());

  }
  cap.release();
  video.release();

  delete[] FCResult;
  delete[] imageInputs;
  delete[] softmax;
}

/*
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char* argv[]) {
  // Check args
  if (argc != 2) {
    cout << "Usage of resnet50 demo: ./resnet50 [model_file]" << endl;
    return -1;
  }
  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "resnet50 should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  /*create runner*/
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
  /*get in/out tensor*/
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  /*get in/out tensor shape*/
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  /*run with batch*/
  runResnet50(runner.get());
  return 0;
}
