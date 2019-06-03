#include "header.h"
#include "utils.hpp"
#include "ParameterManager.hpp"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>


using namespace pcl;

std::string CFG_PARAM_PATH = "/home/inaho-00/work/cpp/denoising_projection/cfg/recognition_parameter.toml";

int
main (int argc, char** argv)
{
    ParameterManager cfg_param(CFG_PARAM_PATH);
    float sigma = cfg_param.ReadFloatData("Param", "sigma");
    float param_R = cfg_param.ReadFloatData("Param", "param_R");
    int param_K = cfg_param.ReadIntData("Param", "param_K");
    std::string DATA_PATH = cfg_param.ReadStringData("Param", "data_path");

    float fx = cfg_param.ReadFloatData("Param", "fx");
    float fy = cfg_param.ReadFloatData("Param", "fy");
    float cx = cfg_param.ReadFloatData("Param", "cx");
    float cy = cfg_param.ReadFloatData("Param", "cy");
    int img_w = cfg_param.ReadIntData("Param", "image_width");
    int img_h = cfg_param.ReadIntData("Param", "image_height");
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (DATA_PATH, *cloud) == -1) 
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    cloud = removeNan(cloud);
    cloud_raw = removeNan(cloud);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    pcl::PointIndices::Ptr sor_inliers(new pcl::PointIndices);
    vector<int> _sor_inliers;
    sor.setInputCloud (cloud);
    sor.setMeanK (30);
    sor.setStddevMulThresh (0.5);
    sor.filter (_sor_inliers);
    sor_inliers->indices = _sor_inliers;

    pcl::ExtractIndices<pcl::PointXYZ> eifilter; // Initializing with true will allow us to extract the removed indices
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (sor_inliers);
    eifilter.setNegative (false);
    eifilter.filter (*cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_noise (new pcl::PointCloud<pcl::PointXYZ>);
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (sor_inliers);
    eifilter.setNegative (true);
    eifilter.filter (*cloud_noise);

    cout << cloud->points.size() << endl;
    cout << cloud_noise->points.size() << endl;
    cout << cloud_raw->points.size() << endl;
    arma::mat output;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_denoised (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised (new pcl::PointCloud<pcl::PointXYZ>);
       
    //std::vector<std::vector<int>> nearestIdx =  getNeighborIdx(cloud, fx, fy, cx, cy, img_w, img_h, rect);
    //calculatePointCloudLocalPlaneGrid(cloud, nearestIdx, output, param_K, param_R, sigma);
    //denoiseWMP(cloud, output, param_K, _cloud_denoised, sigma);

//    nearestIdx =  getNeighborIdx(_cloud_denoised, fx, fy, cx, cy, img_w, img_h, rect);
    //calculatePointCloudLocalPlane(cloud, output, param_K, param_R, sigma);
    //denoiseWMP(cloud, output, param_K, cloud_denoised, sigma);
    //cloud = removeNan(cloud_denoised);
//    cloud_denoised->points.clear();
    //calculatePointCloudLocalPlane(cloud, output, param_K, param_R, sigma);
    //denoiseWMPwithRaw(cloud_noise, cloud, output, param_K, cloud_denoised, sigma);
    //denoiseWMPwithRaw(cloud_noise, cloud, output, param_K, cloud_denoised, sigma);

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    denoisingProjection(cloud_noise, cloud, param_K, cloud_denoised, sigma);
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << "elapsed:" << elapsed << endl;



    cloud_noise = removeNan(cloud_denoised);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    for(int m=0;m<cloud_noise->points.size();m++){
        cloud_out->points.push_back(cloud_noise->points[m]);
    }
    for(int m=0;m<cloud->points.size();m++){
        cloud_out->points.push_back(cloud->points[m]);
    }
    

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    //LocalPlane2RGB(cloud, output, cloud_rgb);
    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis();
    //viewer->addPointCloud<pcl::PointXYZ> (cloud_denoised, "sample cloud");
    viewer->addPointCloud<pcl::PointXYZ> (cloud_out, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");  

    pcl::io::savePCDFileBinary("../data/denoised.pcd", *cloud_out);

    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_rgb);
    //viewer->addPointCloud<pcl::PointXYZRGB> (cloud_rgb, rgb, "sample cloud");
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");  

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return (0);
}