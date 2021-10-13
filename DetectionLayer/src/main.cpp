#include <iostream>
#include <cmath>

#include <stdio.h>
#include <string.h>
#define ANCHS 9
#define GRID_SIZE 7
#define CLASSES 1
#define BOX_PRED (5+CLASSES)
#define CELL_PRED ((BOX_PRED)*ANCHS)
#define DPU_OUTPUT (GRID_SIZE*GRID_SIZE*CELL_PRED)

#define THRESHOLD_IOU 0.5
#define THRESHOLD_SCORE 0.5
#define THRESHOLD_BOXES 100

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
			
			void setBox(int8_t (&predictions)[DPU_OUTPUT], int offset)
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
			}
			

			
			float getObjectness()
			{
				return sigmoid((float) *ob);
			}

			void getClassScores(float (&classes)[CLASSES])
			{
				for(uint8_t i = 0; i < CLASSES; i++)
					classes[i] = sigmoid((float) * cl[i]);
			}

			void setXY(uint8_t &gridx, uint8_t &gridy, float (&anchors)[2])
			{	
				if(eval == true)
				{
					float xy[2] = {sigmoid((float) *tx),sigmoid((float) *ty)};
					float wh[2] = {sigmoid((float) *tw),sigmoid((float) *th)};
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

			void setBoxScore()
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
					float tmpScore = classes[index]*objectness;

						std::cout << (int)index <<std::endl;
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
			void setGridScores()
			{
				for (uint8_t i = 0; i < ANCHS; i++)
					bboxes[i].setBoxScore();
				
			}

			void setCellXY(uint8_t &gridx, uint8_t &gridy, float (&anchors)[ANCHS][2])
			{

				for (uint8_t i = 0; i < ANCHS; i++)
					bboxes[i].setXY(gridx, gridy, anchors[i]);
			}
	};

	class grid
	{	
		private:

		public:

			cell cells[GRID_SIZE][GRID_SIZE];


			grid(int8_t (&predictions)[DPU_OUTPUT])
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
			void setAllScoresXY(float (&anchors)[ANCHS][2])
			{

				for(uint8_t i = 0; i < GRID_SIZE; i++)
					for(uint8_t j = 0; j < GRID_SIZE; j++)
					{
						cells[i][j].setGridScores();
						cells[i][j].setCellXY(i,j, anchors);
					}
			}
			

			
			void doGridNMS(float (&anchors)[ANCHS][2])
			{
				
				setAllScoresXY(anchors);
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

			void gridPrint(float (&anchors)[ANCHS][2])
			{
				doGridNMS(anchors);
				
				for (uint8_t i = 0; i < GRID_SIZE; i++)
					for (uint8_t j = 0; j < GRID_SIZE; j++)
						for (uint8_t p = 0; p < ANCHS; p++)
							if (cells[i][j].bboxes[p].eval == true)
								std::cout << "True box detected at: [" << (int)i << "][" << (int)j << "][" << (int)p << "]" << std::endl;
			}

	};
}
using namespace detection;
int main(int argc, char *argv[])
{

	int8_t tmp[DPU_OUTPUT] = {};

	grid grid(tmp);
	float nanks[ANCHS][2] = {{0.2,0.2},{0.2,0.2},{0.2,0.2},{0.2,0.2},{0.2,0.2},{0.2,0.2},{0.2,0.2},{0.2,0.2},{0.2,0.2}};
	grid.gridPrint(nanks);
	//int a = 1;
	//std::cout << a;
//
	//int *b;
	//b = &a;
	//*b = 2;
	//std::cout << a;


	for(uint8_t i = 0; i < 6; i++)
		tmp[6*54*7 + 3*54 + 5*6 + i] = (int8_t) 4;

	for(uint8_t i = 0; i < 6; i++)
		tmp[6*54*7 + 6*54 + 3*6 + i] = (int8_t) 21;

	grid.gridPrint(nanks);
	//for(uint8_t i = 0; i < 6; i++)
	//	std::cout << (float)tmp[6*54*7 + 3*54 + 5*6 + i];
	float b1[4] = {1,1,2,2};
	float b2[4] = {1+1.1,1+.5,2+.5,2+.5};
	std::cout << (float)grid.cells[6][3].bboxes[5].score;

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