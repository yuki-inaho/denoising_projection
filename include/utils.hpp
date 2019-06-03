#pragma once

#include "header.h"
#include <boost/thread/thread.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/pcl_search.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <forward_list>
#include <iomanip>
#include <fstream>

#include <forward_list> 
#include <algorithm>  // std::sort, std::unique
#include <unordered_map>   
#include <map> 
#include <omp.h>


using namespace std;
using namespace pcl;

double D_MAX = std::numeric_limits<double>::max();
double D_MIN = std::numeric_limits<double>::min();

//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<float> 
getSimilarityVector(const pcl::PointCloud<pcl::PointXYZ> &cloud, const pcl::PointXYZ center_point, float sigma){
    std::vector<float> similarity_vec;
    for(int i=0; i<cloud.points.size(); i++){
        if(std::isfinite(cloud.points[i].x)){
            float sqr_diff_x, sqr_diff_y, sqr_diff_z;
            sqr_diff_x = (cloud.points[i].x - center_point.x) * (cloud.points[i].x - center_point.x);
            sqr_diff_y = (cloud.points[i].y - center_point.y) * (cloud.points[i].y - center_point.y);
            sqr_diff_z = (cloud.points[i].z - center_point.z) * (cloud.points[i].z - center_point.z);
            float diff = sqr_diff_x + sqr_diff_y + sqr_diff_z;
            similarity_vec.push_back(exp(- diff/(2*sigma*sigma))); 
        }else{
            similarity_vec.push_back(0);
        }
    }

    float similarity_sum = 0;
    for(int j=0;j<similarity_vec.size();j++){
        similarity_sum += similarity_vec[j];
    }

    for(int j=0;j<similarity_vec.size();j++){
        similarity_vec[j] /= similarity_sum;
    }

    return similarity_vec;
}



std::vector<float> 
getDiffVector(const pcl::PointCloud<pcl::PointXYZ> &cloud, const pcl::PointXYZ center_point){
    std::vector<float> diff_vec;

    for(int i=0; i<cloud.points.size(); i++){
        if(std::isfinite(cloud.points[i].x)){
            float sqr_diff_x, sqr_diff_y, sqr_diff_z;
            sqr_diff_x = (cloud.points[i].x - center_point.x) * (cloud.points[i].x - center_point.x);
            sqr_diff_y = (cloud.points[i].y - center_point.y) * (cloud.points[i].y - center_point.y);
            sqr_diff_z = (cloud.points[i].z - center_point.z) * (cloud.points[i].z - center_point.z);
            float diff = sqr_diff_x + sqr_diff_y + sqr_diff_z;
            diff_vec.push_back(diff); 
        }else{
            diff_vec.push_back(1000000);
        }
    }

    return diff_vec;
}

pcl::PointCloud<pcl::PointXYZ>
denoisingProjection(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_guide, int _K, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised, float sigma){

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud_guide);

    int K = _K;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    int threads_ = 8;

/*
    pcl::PointCloud<pcl::PointXYZ> _cloud;
    for (int idx = 0; idx < static_cast<int> (cloud_input->points.size ()); ++idx){
        pcl::PointXYZ _point;
        _cloud.points.push_back(_point);
    }
    */
    
    forward_list<pcl::PointXYZ> _point_list;
    #pragma omp parallel for shared (_point_list) private (pointIdxNKNSearch, pointNKNSquaredDistance) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud_input->points.size ()); ++idx)
    {
        if ( kdtree.nearestKSearch (cloud_input->points[idx], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            pcl::PointXYZ center_point = cloud_input->points[idx];
            std::vector<float> similarity_vec;
            std::vector<float> diff_vec;
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr __cloud (new pcl::PointCloud<pcl::PointXYZ>);
            //for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){                
            for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i){
                pcl::PointXYZ point;
                point.x = cloud_guide->points[pointIdxNKNSearch[i]].x;
                point.y = cloud_guide->points[pointIdxNKNSearch[i]].y;
                point.z = cloud_guide->points[pointIdxNKNSearch[i]].z;
                __cloud->points.push_back(point);
            }

            similarity_vec = getSimilarityVector(*__cloud, center_point, sigma);
            diff_vec = getDiffVector(*__cloud, center_point);
            float sim_diff_rate_sum = 0;
            for (size_t i = 1; i < pointIdxNKNSearch.size (); ++i){
                sim_diff_rate_sum += similarity_vec[i] /(diff_vec[i] + 0.000001) ;
            }
            pcl::PointXYZ denoised_point;
            denoised_point.x = 0; denoised_point.y = 0; denoised_point.z = 0;
            for (size_t i = 1; i < pointIdxNKNSearch.size (); ++i){
                denoised_point.x += similarity_vec[i] /(diff_vec[i] + 0.000001) * __cloud->points[i].x / sim_diff_rate_sum;
                denoised_point.y += similarity_vec[i] /(diff_vec[i] + 0.000001) * __cloud->points[i].y / sim_diff_rate_sum;
                denoised_point.z += similarity_vec[i] /(diff_vec[i] + 0.000001) * __cloud->points[i].z / sim_diff_rate_sum;
            }

            //cloud_denoised->points[idx] = denoised_point;
            _point_list.push_front(denoised_point);
        }
    }

    cloud_denoised->points.clear();
    for(auto it=_point_list.begin(); it!=_point_list.end(); ++it){
        pcl::PointXYZ _point;
        _point.x = it->x;
        _point.y = it->y;
        _point.z = it->z;

        cloud_denoised->points.push_back(_point);
    }
    /*
    for (int idx = 0; idx < static_cast<int> (cloud_input->points.size ()); ++idx)
    {
        pcl::PointXYZ _point;
        _point.x = _cloud.points[idx].x;        
        _point.y = _cloud.points[idx].y;        
        _point.z = _cloud.points[idx].z;        
        cloud_denoised->points.push_back(_point);
    }
    */

    cout << cloud_denoised->points.size() << endl;
}


pcl::visualization::PCLVisualizer::Ptr simpleVis ()
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

//removeNan: NaN要素を点群データから除去するメソッド
//input : target(NaN要素を除去する対象の点群)
//output: cloud(除去を行った点群)
pcl::PointCloud<PointXYZ>::Ptr removeNan(pcl::PointCloud<pcl::PointXYZ>::Ptr target){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  int n_point = target->points.size();
  for(int i=0;i<n_point; i++){
    pcl::PointXYZ tmp_point;
    if(std::isfinite(target->points[i].x) && std::isfinite(target->points[i].y) && std::isfinite(target->points[i].z)){
    if(target->points[i].z > 0){
      tmp_point.x = target->points[i].x;
      tmp_point.y = target->points[i].y;
      tmp_point.z = target->points[i].z;
      cloud->points.push_back(tmp_point);
    }
    }
  }
//  cout << "varid points:" << cloud->points.size() << endl;
  return cloud;
}
