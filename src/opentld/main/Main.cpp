/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * MainX.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include "Main.h"

#include "Config.h"
#include "ImAcq.h"
#include "Gui.h"
#include "TLDUtil.h"
#include "Trajectory.h"
#include "Timing.h"
#include "iostream"/////////////////////////////////


using namespace tld;
using namespace cv;
using namespace std;/////////////////////////////////

void Main::doWork()
{
    Trajectory trajectory;
    IplImage *img = imAcqGetImg(imAcq);
    Mat grey(img->height, img->width, CV_8UC1);//初始化一个灰度图像
    cv::Mat img2 = cv::cvarrToMat(img); ///////////////////////////////////
    cvtColor(img2, grey, CV_BGR2GRAY);//原图转化为灰度图
    //cvtColor(img, grey, CV_BGR2GRAY);

    tld->detectorCascade->setImgSize(grey.cols, grey.rows, grey.step);

#ifdef CUDA_ENABLED
    tld->learningEnabled = false;
    selectManually = false;

    if(tld->learningEnabled || selectManually)
        std::cerr << "Sorry. Learning and manual object selection is not supported with CUDA implementation yet!!!" << std::endl;
#endif

	if(showTrajectory) //use point to draw trajectory?????
	{
		trajectory.init(trajectoryLength);
	}

    if(selectManually)
    {

        CvRect box;

        if(getBBFromUser(img, box, gui) == PROGRAM_EXIT) //手动画框
        {
            return;
        }

        if(initialBB == NULL)
        {
            initialBB = new int[4];
        }

        initialBB[0] = box.x;
        initialBB[1] = box.y;
        initialBB[2] = box.width;
        initialBB[3] = box.height;
    }


    FILE *resultsFile = NULL; //用于打开文件储存结果

    if(printResults != NULL)
    {
        resultsFile = fopen(printResults, "w");
    }

    bool reuseFrameOnce = false;//？？？？？？？？？？？？？？？？？？？？不懂
    bool skipProcessingOnce = false;

    if(loadModel && modelPath != NULL)// 读取模型且能找到
    {
        tld->readFromFile(modelPath);
        reuseFrameOnce = true;
    }
    else if(initialBB != NULL)
    {
        Rect bb = tldArrayToRect(initialBB);

        printf("Starting at %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);

        tld->selectObject(grey, &bb);// 确定模型进行跟踪？？！！！！！！！！
        skipProcessingOnce = true;
        reuseFrameOnce = true;
    }
    while(imAcqHasMoreFrames(imAcq))//还有后续图像，即摄像头还开着，就保持循环
    {
        tick_t procInit, procFinal;
        double tic = cvGetTickCount(); // 返回该处代码执行所耗的时间，单位为秒


        if(!reuseFrameOnce)
        {
            img = imAcqGetImg(imAcq);

            if(img == NULL)
            {
                printf("current image is NULL, assuming end of input.\n");
                break;
            }
    	    cv::Mat img2 = cv::cvarrToMat(img); ///////////////////////////////////
            cvtColor(img2, grey, CV_BGR2GRAY);//////////////////////////
            //cvtColor(img, grey, CV_BGR2GRAY);
        }

        if(!skipProcessingOnce)
        {
            getCPUTick(&procInit);
            tld->processImage(img2); //整个跟踪监测学习的过程在这里调用
            getCPUTick(&procFinal);
            PRINT_TIMING("FrameProcTime", procInit, procFinal, "\n");
        }
        else
        {
            skipProcessingOnce = false;
        }

        if(printResults != NULL)
        {
            if(tld->currBB != NULL)
            {
                fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame - 1, tld->currBB->x, tld->currBB->y, tld->currBB->width, tld->currBB->height, tld->currConf); // 在屏幕输出跟踪位置
            }
            else
            {
                fprintf(resultsFile, "%d NaN NaN NaN NaN NaN\n", imAcq->currentFrame - 1);
            }
        }

        double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

        toc = toc / 1000000;

        float fps = 1 / toc; // calculate fps

        int confident = (tld->currConf >= threshold) ? 1 : 0; // 置信度是否大于设定的阈值


        if(showOutput || saveDir != NULL)
        {
            char string[128];

            char learningString[10] = "";

            if(tld->learning)
            {
                strcpy(learningString, "Learning");
            }

            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, tld->currConf, fps, tld->detectorCascade->numWindows, learningString);
            CvScalar yellow = CV_RGB(255, 255, 0); // 定义几种颜色
            CvScalar blue = CV_RGB(0, 0, 255);
            CvScalar black = CV_RGB(0, 0, 0);
            CvScalar white = CV_RGB(255, 255, 255);


            if(tld->currBB != NULL)
            {

                //cout << "currBB is not NULL" << endl;//test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                //cout << "x=" << tld->currBB->x << "y=" << tld->currBB->y << endl;//test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1




                CvScalar rectangleColor = (confident) ? blue : yellow; // 确信度高蓝色，低黄色？？
                cvRectangle(img, tld->currBB->tl(), tld->currBB->br(), rectangleColor, 8, 8, 0); // 长方形绘图函数！！！！！！！！！！！！！！

				if(showTrajectory)
				{
					CvPoint center = cvPoint(tld->currBB->x+tld->currBB->width/2, tld->currBB->y+tld->currBB->height/2);
					cvLine(img, cvPoint(center.x-2, center.y-2), cvPoint(center.x+2, center.y+2), rectangleColor, 2);
					cvLine(img, cvPoint(center.x-2, center.y+2), cvPoint(center.x+2, center.y-2), rectangleColor, 2);
					trajectory.addPoint(center, rectangleColor);
				}
            }
			else if(showTrajectory)
			{
				trajectory.addPoint(cvPoint(-1, -1), cvScalar(-1, -1, -1));
			}

			if(showTrajectory)
			{
				trajectory.drawTrajectory(img);
			}

            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8);
//            cvRectangle(img, cvPoint(0, 0), cvPoint(img->width, 50), black, CV_FILLED, 8, 0);
            cvPutText(img, string, cvPoint(25, 25), &font, white);

            if(showForeground)
            {

                for(size_t i = 0; i < tld->detectorCascade->detectionResult->fgList->size(); i++)
                {
                    Rect r = tld->detectorCascade->detectionResult->fgList->at(i);
                    cvRectangle(img, r.tl(), r.br(), white, 1);
                }

            }


            if(showOutput)
            {


                CvSize size = cvSize(img->width*2,img->height*2);
                IplImage*img2 =cvCreateImage(size,img->depth,img->nChannels);
                cvResize(img, img2,CV_INTER_LINEAR);



                //gui->showImage(img2); // ！！！！！！！！！！！！！！！！！！！！！！
                cvShowImage("TLD", img2);
                //char key = gui->getKey(); // ！！！！！！！！！！！！！！！！！
                char key = cvWaitKey(10);

                if(key == 'q') break;

                /*
                if(key == 'b')
                {

                    ForegroundDetector *fg = tld->detectorCascade->foregroundDetector;

                    if(fg->bgImg.empty())
                    {
                        fg->bgImg = grey.clone();
                    }
                    else
                    {
                        fg->bgImg.release();
                    }
                }
                */

                if(key == 'c')
                {
                    //clear everything
                    tld->release();
                }

                if(key == 'l')
                {
                    tld->learningEnabled = !tld->learningEnabled;
                    printf("LearningEnabled: %d\n", tld->learningEnabled);
                }

                if(key == 'a')
                {
                    tld->alternating = !tld->alternating;
                    printf("alternating: %d\n", tld->alternating);
                }

                if(key == 'e')
                {
                    tld->writeToFile(modelExportFile);
                }

                if(key == 'i')
                {
                    tld->readFromFile(modelPath);
                }

                if(key == 'r')
                {
                    CvRect box;

                    if(getBBFromUser(img, box, gui) == PROGRAM_EXIT)
                    {
                        break;
                    }

                    Rect r = Rect(box);

                    tld->selectObject(grey, &r);
                }
            }

            if(saveDir != NULL)
            {
                char fileName[256];
                sprintf(fileName, "%s/%.5d.png", saveDir, imAcq->currentFrame - 1);

                cvSaveImage(fileName, img);
            }
        }

        if(!reuseFrameOnce)
        {
            cvReleaseImage(&img);
        }
        else
        {
            reuseFrameOnce = false;
        }
    }

    if(exportModelAfterRun)
    {
        tld->writeToFile(modelExportFile);
    }
}
