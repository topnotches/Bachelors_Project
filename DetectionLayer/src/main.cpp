#include <iostream>
#include <cmath>

#include <stdio.h>
#include <string.h>
#define ANCHS 9
#define GRID_SIZE 7
#define CLASSES 2
#define BOX_PRED (5+CLASSES)
#define CELL_PRED ((BOX_PRED)*ANCHS)
#define DPU_OUTPUT (GRID_SIZE*GRID_SIZE*CELL_PRED)

#define THRESHOLD_IOU 0.5
#define THRESHOLD_SCORE 0.5

namespace detection
{ 

	class bbox
	{
		private:
			int8_t *tx;
			int8_t *ty;
			int8_t *tw;
			int8_t *th;
			int8_t *ob;
			int8_t *cl[CLASSES];

			float score;
			uint8_t index;


		public:
			bool eval = false;
			void setBox(int8_t (&predictions)[DPU_OUTPUT], int offset)
			{
				tx = &predictions[offset];
				ty = &predictions[offset + 1];
				tw = &predictions[offset + 2];
				th = &predictions[offset + 3];
				ob = &predictions[offset + 4];
				for (uint8_t i = 0; i < CLASSES; i++)
					cl[i] = &predictions[offset + 5 + 1];
			}
			void printPredBox()
			{
				std::cout << (int) *tx << (int) *ty << (int) *tw << (int) *th << (int) *ob;
				
				for (uint8_t i = 0; i < CLASSES; i++)
					std::cout << (int) *cl[i];
			}
			

			float sigmoid(float x)
			{
				return 1/(1+exp(-x));
			}
			
			float getObjectness()
			{
				return sigmoid((float) * ob);
			}

			void getClassScores(float (&classes)[CLASSES])
			{
				
				for(uint8_t i = 0; i < CLASSES; i++)
					classes[i] = sigmoid((float) * cl[i]);
			}

			void setXY(float (&xyCorners)[4], uint8_t gridx, uint8_t gridy)
			{	
				float xy[2] = {sigmoid((float) *tx),sigmoid((float) *ty)};
				float wh[2] = {sigmoid((float) *tw),sigmoid((float) *th)};
				xy[0] = (xy[0] + ((float) gridx))/GRID_SIZE;
				xy[1] = (xy[1] + ((float) gridy))/GRID_SIZE;
				xyCorners[0] = xy[0] - wh[0]/2;
				xyCorners[1] = xy[1] - wh[1]/2;
				xyCorners[2] = xy[0] + wh[0]/2;
				xyCorners[3] = xy[1] + wh[1]/2;
			}
			float getScore()
			{
				float objectness = getObjectness();
				if (objectness>THRESHOLD_SCORE)
				{

					float classes[CLASSES];
					getClassScores(classes);
					index = 0;
					for (uint8_t i = 0; i < CLASSES; i++)
					{
						if(classes[i]>classes[index])
							index = i;
					}
					
					if (classes[index] > THRESHOLD_SCORE)
						eval = true;
					else
						eval = false;
					
				}
				else
					eval = false;
				
				
			}

	};
	
	float getIOU(float xy1[4], float xy2[4])
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

	class cell
	{
		private:
			bbox bboxes[ANCHS];
		
		public:
			void setCellBoxes(int8_t (&predictions)[DPU_OUTPUT], int offset)
			{
				for(unsigned int i = 0; i < ANCHS; i++)
					bboxes[i].setBox(predictions, i*BOX_PRED + offset);
			}
			void printPredCell()
			{
				for(unsigned int i = 0; i < ANCHS; i++)
					bboxes[i].printPredBox();
			}
	};
	class grid
	{	private:
			cell grid[GRID_SIZE][GRID_SIZE];
		
		public:
			void setGridCells(int8_t (&predictions)[DPU_OUTPUT])
			{
				for(unsigned int i = 0; i < GRID_SIZE; i++)
					for(unsigned int j = 0; j < GRID_SIZE; j++)
						grid[i][j].setCellBoxes(predictions, i*54*7 + j*54);
			}
			
			void printPredGrid()
			{
				for(unsigned int i = 0; i < GRID_SIZE; i++)
					for(unsigned int j = 0; j < GRID_SIZE; j++)
						grid[i][j].printPredCell();
			}

	};
}
using namespace detection;

int main(int argc, char *argv[])
{

	//int8_t tmp[DPU_OUTPUT] = {};
//
//
	//grid grid;
	//grid.setGridCells(tmp);
	//bbox box;
	//std::cout<<box.sigmoid(0);

	//int a = 1;
	//std::cout << a;
//
	//int *b;
	//b = &a;
	//*b = 2;
	//std::cout << a;


	float b1[4] = {1,1,2,2};
	float b2[4] = {1+.5,1+.5,2+.5,2+.5};
	std::cout << "IOU:   " << getIOU(b1,b2);

	//for (unsigned int i = 0; i < GRID_SIZE; i++)
	//	for (unsigned int j = 0; j < GRID_SIZE; j++)
	//		for (unsigned int b = 0; b < ANCHS; b++)
	//		{
	//			tmp[i*54*7 + j*54 + b*6] = (int8_t)0;
	//			tmp[i*54*7 + j*54 + b*6 + 1] = (int8_t)0;
	//			tmp[i*54*7 + j*54 + b*6 + 2] = (int8_t)i;
	//			tmp[i*54*7 + j*54 + b*6 + 3] = (int8_t)j;
	//			tmp[i*54*7 + j*54 + b*6 + 4] = (int8_t)b;
	//		}
//
//
	//grid.printPredGrid();
//
	//grid.cell
//
	//for(unsigned int i; i < DPU_OUTPUT; i++)
	//	std::cout<<(int)tmp[i];

	
	return 0;
}