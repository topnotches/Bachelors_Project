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
#define THRESHOLD_SCORE 0.8
#define THRESHOLD_BOXES 100
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
				
				for (uint8_t i = 0; i < GRID_SIZE; i++)
					for (uint8_t j = 0; j < GRID_SIZE; j++)
						for (uint8_t p = 0; p < ANCHS; p++)
							if (cells[i][j].bboxes[p].eval == true)
              {
                cv::Point pt1(cells[i][j].bboxes[p].xy12[0]*227, cells[i][j].bboxes[p].xy12[1]*227);
                cv::Point pt2(cells[i][j].bboxes[p].xy12[2]*227, cells[i][j].bboxes[p].xy12[3]*227);
                cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
              }
      }

	};
}
using namespace detection;
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
void runResnet50(vart::Runner* runner) {
  /* Mean value for ResNet50 specified in Caffe prototxt */
  vector<string> kinds, images;

  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  /* Load all kinds words.*/
  LoadWords(wordsPath + "words.txt", kinds);
  if (kinds.size() == 0) {
    cerr << "\nError: No words exist in file words.txt." << endl;
    return;
  }
  float mean[3] = {104, 107, 123};

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

  int batchSize = in_dims[0];

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  vector<Mat> imageList;
  int8_t* imageInputs = new int8_t[inSize * batchSize];

  float* softmax = new float[outSize];
  int8_t* FCResult = new int8_t[batchSize * outSize];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  grid grid;
  grid.grod(FCResult);


  float nanks[ANCHS][2] = {{0.02403846, 0.03125   },
      {0.03846154, 0.07211538},
      {0.07932692, 0.05528846},
      {0.07211538, 0.14663461},
      {0.14903846, 0.10817308},
      {0.14182693, 0.28605768},
      {0.27884614, 0.21634616},
      {0.375     , 0.47596154},
      {0.89663464, 0.78365386}};
  /*run with batch*/
  for (unsigned int n = 0; n < images.size(); n += batchSize) {
    unsigned int runSize =(images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;
    in_dims[0] = runSize;
    out_dims[0] = batchSize;
    Mat image = imread("../images/001.jpg");

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
    
    grid.drawRectangles(image, nanks, output_scale);
    grid.gridPrint( nanks, output_scale);
    imwrite("res.jpg", image);


    imageList.clear();
    inputs.clear();
    outputs.clear();
  }
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
