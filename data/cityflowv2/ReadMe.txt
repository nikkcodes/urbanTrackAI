%******************************************************************************************************************%
% The AIC22 benchmark (CityFlowV2) is captured by 46 cameras in real-world traffic surveillance environment.       %
% A total of 880 vehicles are annotated in 6 different scenarios. 3 of the scenarios are used for training.        %
% 2 scenarios are for validation. And the rest of the scenario is for testing                                      %
% There are 215.03 minutes of videos in total. The length of the training videos is 58.43 minutes, the validation  %
% videos 136.60 minutes, and the testing videos 20.00 minutes.                                                     %
%******************************************************************************************************************%

Content in the directory:
1. "train/*". It contains all the subsets for training. 
2. "validation/*". It contains all the subsets for validation. 
3. "test/*". It contains the subset for testing.
4. "train(validation/test)/<subset>/<cam>/vdo.avi". They are the test videos. 
5. "train(validation/test)/<subset>/<cam>/roi.jpg". They are the region of interest (ROI), where the white area covers the entire body of annotated vehicle targets. 
6. "train(validation)/<subset>/<cam>/gt/gt.txt". They list the ground truths of MTMC tracking in the MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1]. Only vehicles that pass through at least 2 cameras are taken into account. 
7. "train(validation/test)/<subset>/<cam>/det/det_*.txt". They are the detection results from different baselines in the MOTChallenge format [frame, -1, left, top, width, height, conf, -1, -1, -1]. The reference for each baseline method is given as follows.
[YOLOv3] Redmon, Joseph and Farhadi, Ali, "YOLOv3: An Incremental Improvement," arXiv, 2018.
[SSD] Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C., "SSD: Single Shot MultiBox Detector," ECCV, 2016.
[Mask/Faster R-CNN] He, Kaiming and Gkioxari, Georgia and Dollár, Piotr and Girshick, Ross, "Mask R-CNN," ICCV, 2017.
8. "train(validation/test)/<subset>/<cam>/mtsc/mtsc_*.txt". They are the MTSC tracking results from different baselines in the MOTChallenge format [frame, ID, left, top, width, height, 1, -1, -1, -1]. The reference for each baseline method is given as follows.
[Deep SORT] Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich, "Simple Online and Realtime Tracking with a Deep Association Metric," ICIP, 2017.
[Tracklet Clustering] Tang, Zheng and Wang, Gaoang and Xiao, Hao and Zheng, Aotian and Hwang, Jenq-Neng, "Single-camera and Inter-camera Vehicle Tracking and 3D Speed Estimation Based on Fusion of Visual and Semantic Features," CVPRW, 2018.
[MOANA] Tang, Zheng and Hwang, Jenq-Neng, "MOANA: An Online Learned Adaptive Appearance Model for Robust Multiple Object Tracking in 3D," IEEE Access, 2019.
[TNT] Hsu, Hung-Min and Huang, Tsung-Wei and Wang, Gaoang and Cai Jiarui and Lei, Zhichao and Hwang, Jenq-Neng, "Multi-Camera Tracking of Vehicles based on Deep Features Re-ID and Trajectory-Based Camera Link Models," CVPRW, 2019.
9. "train(validation/test)/<subset>/<cam>/segm/segm_mask_rcnn.txt". They are the segmentation results from Mask R-CNN (each line corresponds to the detection results in det_mask_rcnn.txt). 
10. "train(validation/test)/<subset>/<cam>/calibration.txt". They are the baseline manual calibration results. Each file shows the 3x3 homography matrix at the first line. If the correction of radial distortion is conducted, the 3x3 intrinsic parameter matrix and 1x4 distortion coefficients are also printed. Finally, the reprojection error in pixels is printed as well.
11. "list_cam.txt". It lists the subfolder of each video for training/validation/testing.
12. "cam_loc/<subset>.png". They are the maps with the camera locations. Since we do not have access to the exact GPS location of each camera, the GPS location for the approximate center of each scenario is provided. 
The GPS location for S01.png is 42.525678, -90.723601.
The GPS location for S02.png is 42.491916, -90.723723.
The GPS location for S0345.png is 42.498780, -90.686393.
The GPS location for S06.png is 42.492448, -90.723343.
13. "cam_timestamp/<subset>.txt". They list the (starting) timestamps of videos in seconds for each of the 6 scenarios. Note that due to noise in video transmission, which is common in real deployed systems, some frames are skipped or stuck within some videos, so they are not perfectly aligned. The frame rates of all the videos are 10 FPS, except for c015 in S03, whose frame rate is 8 FPS. 
14. "cam_framenum/<subset>.txt". They list the numbers of video frames for each of the 6 scenarios. 
15. "eval/". This is the evaluation code for MTMC tracking. It can be executed via "python3 eval.py <ground_truth> <prediction> --dstype <dstype>". Please install the required packages when running in Python.  

Citations: 

@inproceedings{Tang19CityFlow,  
author = {Zheng Tang and Milind Naphade and Ming-Yu Liu and Xiaodong Yang and Stan Birchfield and Shuo Wang and Ratnesh Kumar and David Anastasiu and Jenq-Neng Hwang},  
title = {City{F}low: {A} city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification},  
booktitle = {Proc. CVPR},  
pages = {8797--8806},  
address = {Long Beach, CA, USA},  
year = {2019}  
}

@inproceedings{Naphade19AIC19,  
author = {Milind Naphade and Zheng Tang and Ming-Ching Chang and David C. Anastasiu and Anuj Sharma and Rama Chellappa and Shuo Wang and Pranamesh Chakraborty and Tingting Huang and Jenq-Neng Hwang and Siwei Lyu},  
title = {The 2019 {AI} {C}ity {C}hallenge},  
booktitle = {Proc. CVPR Workshops},  
pages = {452--460},  
address = {Long Beach, CA, USA},  
year = {2019}  
}

@inproceedings{Naphade20AIC20,  
author = {Milind Naphade and Shuo Wang and David C. Anastasiu and Zheng Tang and Ming-Ching Chang and Xiaodong Yang and Liang Zheng and Anuj Sharma and Rama Chellappa and Pranamesh Chakraborty},  
title = {The 4th {AI} {C}ity {C}hallenge},  
booktitle = {Proc. CVPR Workshops},
address = {Virtual},  
year = {2020}  
}

@inproceedings{Naphade21AIC21,
author = {Milind Naphade and Shuo Wang and David C. Anastasiu and Zheng Tang and Ming-Ching Chang and Xiaodong Yang and Yue Yao and Liang Zheng and Pranamesh Chakraborty and Christian E. Lopez and Anuj Sharma and Qi Feng and Vitaly Ablavsky and Stan Sclaroff},
title = {The 5th {AI} {C}ity {C}hallenge},
booktitle = {Proc. CVPR Workshops},
pages = {4263--4273},
address = {Virtual},
year = {2021}
}

If you have any question, please contact aicitychallenges@gmail.com.
