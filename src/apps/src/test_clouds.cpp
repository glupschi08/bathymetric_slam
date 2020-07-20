/* Copyright 2019 Ignacio Torroba (torroba@kth.se)
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>
#include <vector>
#include <cmath>

#include "data_tools/std_data.h"
#include "data_tools/benchmark.h"

#include "submaps_tools/cxxopts.hpp"
#include "submaps_tools/submaps.hpp"

#include "registration/utils_visualization.hpp"
#include "registration/gicp_reg.hpp"

#include "graph_optimization/utils_g2o.hpp"
#include "graph_optimization/graph_construction.hpp"
#include "graph_optimization/ceres_optimizer.hpp"
#include "graph_optimization/read_g2o.h"

#include "bathy_slam/bathy_slam.hpp"

#include <pcl/common/common.h>
#include <pcl/filters/uniform_sampling.h>

//------for keypoint example
#include <Eigen/Core>
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include "pcl/io/pcd_io.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/features/normal_3d.h"
#include "pcl/features/pfh.h"
#include "pcl/keypoints/sift_keypoint.h"
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
//-----update
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <fstream> //write to file


//include for cluster descriptor and dbscan
#include "submaps_tools/OctreeGenerator.h"
#include <math.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include <pcl/common/common.h>
#include <queue>
#include "submaps_tools/dbScan.h"
#include "submaps_tools/keypointcluster.h"


#define SUBMAPS 0
#define FULLMAP 1

// Hyper parameters
#define LEAF_SIZE .1    //for the sampling of the cloud -> 0.1 no filtering applied

//for computing the normals
#define normal_radius 5//1.2 //0.25  -> very good results with 0.25, 5, 10 and feature radius10.25

//  for compute_PFHRGB_features
// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
#define feature_radius 5.25//3.25  //0.25

// Parameters for sift computation
/*
#define min_scale 0.2
#define nr_octaves 4
#define nr_scales_per_octave 5
#define min_contrast 0.25
 */




using namespace Eigen;
using namespace std;
using namespace g2o;

std::tuple<uint8_t, uint8_t, uint8_t> jet(double x)
{
    const double rone = 0.8;
    const double gone = 1.0;
    const double bone = 1.0;
    double r, g, b;

    x = (x < 0 ? 0 : (x > 1 ? 1 : x));

    if (x < 1. / 8.) {
        r = 0;
        g = 0;
        b = bone * (0.5 + (x) / (1. / 8.) * 0.5);
    } else if (x < 3. / 8.) {
        r = 0;
        g = gone * (x - 1. / 8.) / (3. / 8. - 1. / 8.);
        b = bone;
    } else if (x < 5. / 8.) {
        r = rone * (x - 3. / 8.) / (5. / 8. - 3. / 8.);
        g = gone;
        b = (bone - (x - 3. / 8.) / (5. / 8. - 3. / 8.));
    } else if (x < 7. / 8.) {
        r = rone;
        g = (gone - (x - 5. / 8.) / (7. / 8. - 5. / 8.));
        b = 0;
    } else {
        r = (rone - (x - 7. / 8.) / (1. - 7. / 8.) * 0.5);
        g = 0;
        b = 0;
    }

    return std::make_tuple(uint8_t(255.*r), uint8_t(255.*g), uint8_t(255.*b));
}


//does the prescalling for jet -> maps z to [0-1]:[1-0] in the area between 0 and threshold
//e.g. points along a linear line in z direction would get be: blue, green, yellow, red, yellow, green, blue, green,...
std::tuple<uint8_t, uint8_t, uint8_t> stacked_jet(double z, double threshold){
    pcl::PointXYZRGB pointrgb;
    std::tuple<uint8_t, uint8_t, uint8_t> colors_rgb;
    double val;
    if(z<=0){
        while(z<0){
            z+=threshold;
        }
    }else{
        while(z>threshold){
            z-=threshold;
        }
    }
    if(z>threshold/2){
        z-=(threshold/2);
        val=-((z/(threshold/2))-1);
    }else{
        val=z/(threshold/2);
    }
    return jet(val);
}

pcl::visualization::PCLVisualizer::Ptr rgbVis (SubmapsVec& submaps_set, int num, bool jet_flag, double jet_stacking_threshold){
    int vp1_;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    float black = 0.0;  // Black
    float white = 1.0 - black;
    viewer->createViewPort (0.0, 0.0, 1.0, 1.0, vp1_);

    unsigned int i = 0;
    PointCloudRGB::Ptr submap_ptr (new PointCloudRGB);
    for(SubmapObj& submap: submaps_set){
        // Find max and min depth in map
        PointT min, max;
        pcl::getMinMax3D(submap.submap_pcl_, min, max);
        std::cout << "Max " << max.getArray3fMap().transpose() << std::endl;
        std::cout << "Min " << min.getArray3fMap().transpose() << std::endl;
        // Normalize and give colors based on z
        for(PointT& pointt: submap.submap_pcl_.points){
            pcl::PointXYZRGB pointrgb;
            pointrgb.x = pointt.x;
            pointrgb.y = pointt.y;
            pointrgb.z = pointt.z;
            std::tuple<uint8_t, uint8_t, uint8_t> colors_rgb;
            if(jet_flag==true) {
                colors_rgb = stacked_jet(pointt.z, jet_stacking_threshold);
            }else {
                colors_rgb = jet((pointt.z - min.z) / (max.z - min.z));
            }
            std::uint32_t rgb = (static_cast<std::uint32_t>(std::get<0>(colors_rgb)) << 16 |
                                 static_cast<std::uint32_t>(std::get<1>(colors_rgb)) << 8 |
                                 static_cast<std::uint32_t>(std::get<2>(colors_rgb)));
            pointrgb.rgb = *reinterpret_cast<float*>(&rgb);
            submap_ptr->points.push_back(pointrgb);
        }
        std::cout << submap_ptr->points.size() << std::endl;
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_h(submap_ptr);
        viewer->addPointCloud(submap_ptr, rgb_h, "gt_cloud_" + std::to_string(i), vp1_);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "gt_cloud_" + std::to_string(i));
        viewer->addCoordinateSystem(3.0, submap.submap_tf_, "gt_cloud_" + std::to_string(i), vp1_);
        i++;
    }

    //viewer->setBackgroundColor (white, white, white, vp1_);
    //viewer->setBackgroundColor (0,0,0);
    viewer->setBackgroundColor (black, black, black, vp1_);

    return (viewer);
}


void compute_PFHRGB_features(pcl::PointCloud <pcl::PointXYZRGB>::Ptr &cloud,
                             pcl::PointCloud <pcl::Normal>::Ptr &normals,
                             pcl::PointCloud <pcl::PointWithScale>::Ptr &keypoints,
                             pcl::PointCloud <pcl::PFHRGBSignature250>::Ptr &descriptors_out) {


    // copy only XYZ data of keypoints for use in estimating features
    pcl::PointCloud <pcl::PointXYZRGB>::Ptr keypoints_xyzrgb(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::copyPointCloud(*keypoints, *keypoints_xyzrgb);

    // Create the PFH estimation class, and pass the input dataset+normals to it
    pcl::PFHRGBEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHRGBSignature250> pfhrgbEstimation;

    pfhrgbEstimation.setInputCloud(keypoints_xyzrgb);
    pfhrgbEstimation.setSearchSurface(cloud); // use all points for analyzing local cloud structure
    pfhrgbEstimation.setInputNormals(normals);
    // alternatively, if cloud is of tpe PointNormal, do pfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the PFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    pfhrgbEstimation.setSearchMethod(tree);
    //pfhrgbEstimation.setKSearch(100);

    // Use all neighbors in a sphere of radius radius
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    pfhrgbEstimation.setRadiusSearch(feature_radius);

    // Compute the features
    pfhrgbEstimation.compute(*descriptors_out);
}

void compute_normals(pcl::PointCloud <pcl::PointXYZRGB>::Ptr &points,
                     pcl::PointCloud <pcl::Normal>::Ptr &normals_out) {

    pcl::NormalEstimation <pcl::PointXYZRGB, pcl::Normal> norm_est;
    // Use a FLANN-based KdTree to perform neighbourhood searches
    norm_est.setSearchMethod(pcl::search::KdTree <pcl::PointXYZRGB>::Ptr(new pcl::search::KdTree <pcl::PointXYZRGB>));

    norm_est.setRadiusSearch(normal_radius);
    norm_est.setInputCloud(points);
    norm_est.compute(*normals_out);
}

/*
void detect_keypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast,
        pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out){


    pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;

    // Use a FLANN-based KdTree to perform neighborhood searches
    sift_detect.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));

    // Set the detection parameters
    sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast (min_contrast);

    // Set the input
    sift_detect.setInputCloud (points);

    // Detect the keypoints and store them in "keypoints_out"
    sift_detect.compute (*keypoints_out);
}
*/
void detect_keypoints(pcl::PointCloud <pcl::PointXYZRGB>::Ptr &points,
                      pcl::PointCloud <pcl::PointWithScale>::Ptr &keypoints_out,
                      float min_scale, int nr_octaves, int nr_scales_per_octave,float min_contrast) {


    std::cout << "min_scale " << min_scale << std::endl;
    std::cout << "nr_octaves " << nr_octaves << std::endl;
    std::cout << "nr_scales_per_octave " << nr_scales_per_octave << std::endl;
    std::cout << "min_contrast " << min_contrast << std::endl;

    pcl::SIFTKeypoint <pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;

    // Use a FLANN-based KdTree to perform neighbourhood searches
    pcl::search::KdTree <pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree <pcl::PointXYZRGB>);
    sift_detect.setSearchMethod(tree);

    // Set the detection parameters
    sift_detect.setScales(min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast(min_contrast);

    // Set the input
    sift_detect.setInputCloud(points);

    // Detect the keypoints and store them in "keypoints.out"
    sift_detect.compute(*keypoints_out);
}




void visualize_keypoints (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points, const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints){
    // Add the points to the vizualizer
    pcl::visualization::PCLVisualizer viz;
    viz.addPointCloud (points, "points");

    // Draw each keypoint as a sphere
    for (size_t i = 0; i < keypoints->size (); ++i)
    {
        // Get the point data
        const pcl::PointWithScale & p = keypoints->points[i];

        // Generate a unique string for each sphere
        std::stringstream ss ("keypoint");
        ss << i;

        // Add a sphere at the keypoint
        viz.addSphere (p, 2*p.scale, 1.0, 0.0, 0.0, ss.str ());
    }

    // Give control over to the visualizer
    viz.spin ();
}


/*
void compute_features(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &tgt,
                                    Eigen::Matrix4f &transform,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr & keypoints_src_visualize_temp,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr & keypoints_tgt_visualize_temp,
                                    pcl::Correspondences & good_correspondences,
                                    std::string keypoints_meth) {*/
void compute_features(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src, pcl::PointCloud<pcl::PointXYZ>::Ptr & keypoints_src_visualize_temp,float min_scale, int nr_octaves, int nr_scales_per_octave,float min_contrast, int show, int octree_resolution, int minPtsAux, int minPts, float eps){

    // ESTIMATING KEY POINTS
    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_src(new pcl::PointCloud<pcl::PointWithScale>);

    cout << "chosen Method is SWIFT" << endl;
    //detect_keypoints(src, keypoints_src);
    detect_keypoints(src, keypoints_src,min_scale,nr_octaves,nr_scales_per_octave,min_contrast);
    cout << "No of SIFT points in the src are " << keypoints_src->points.size() << endl;
    cout << "test 1"  << endl;
/*
    // ESTIMATING PFH FEATURE DESCRIPTORS AT KEYPOINTS AFTER COMPUTING NORMALS
    pcl::PointCloud <pcl::Normal>::Ptr src_normals(new pcl::PointCloud<pcl::Normal>);
    compute_normals(src, src_normals);

    // PFHRGB Estimation
    pcl::PointCloud <pcl::PFHRGBSignature250>::Ptr fpfhs_src_rgb(new pcl::PointCloud<pcl::PFHRGBSignature250>);

    compute_PFHRGB_features(src, src_normals, keypoints_src, fpfhs_src_rgb);
    cout << "End of compute_FPFH_RGB_features! " << endl;
*/
    // Copying the pointwithscale to pointxyz so as visualize the cloud
    pcl::copyPointCloud(*keypoints_src, *keypoints_src_visualize_temp);

    /*
    // Find correspondences between keypoints in FPFH space
    pcl::CorrespondencesPtr all_correspondences_RGB(new pcl::Correspondences);
    findCorrespondences_PFHRGB(fpfhs_src_rgb, fpfhs_tgt_rgb, *all_correspondences_RGB);
    cout << "All correspondences size: " << all_correspondences_RGB->size() << endl;

    rejectBadCorrespondences(all_correspondences_RGB, keypoints_src, keypoints_tgt, good_correspondences);
    //rejectBadCorrespondences(all_correspondences, keypoints_src, keypoints_tgt, good_correspondences);

    cout << "End of rejectBadCorrespondences! " << endl;
    cout << "Good correspondences size: " << good_correspondences.size() << endl;

    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> trans_est;
    trans_est.estimateRigidTransformation(*keypoints_src_visualize_temp, *keypoints_tgt_visualize_temp, good_correspondences, transform);
    */
    if(0 ) {
        pcl::PCDWriter writer;
        pcl::io::savePCDFile("keypoints__visualize_temp.pcd", *keypoints_src_visualize_temp, true);
        pcl::io::savePCDFile("src_cloud.pcd", *src, true);
    }

    //-------------------------------------------------------------------------------------------
    pcl::PointCloud <pcl::PointXYZRGB>::Ptr keypoints_xyzrgb(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::copyPointCloud(*keypoints_src, *keypoints_xyzrgb);


    //new implementaion part
    //do some filtering on the cloud to remove outliers
    if(1){
        // Create the filtering object for RadiusOutlierRemoval
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
        //std::cerr << "setRadiusSearch: " <<test_double1<< std::endl;
        //std::cerr << "setMinNeighborsInRadius: " <<test_double2<< std::endl;
        outrem.setRadiusSearch(5.);//good 5 and r = 3//0.8
        outrem.setMinNeighborsInRadius (4);//2
        std::cout << "keypoints_src_visualize_temp after StatisticalOutlierRemoval: " <<keypoints_xyzrgb->size()<< std::endl;
        outrem.setInputCloud(keypoints_xyzrgb);
        outrem.filter (*keypoints_xyzrgb);
        std::cout << "keypoints_src_visualize_temp after RadiusOutlierRemoval: " <<keypoints_xyzrgb->size()<< std::endl;
    }

    queue<KeypointCluster> Keypoint_Cluster_Queue;
    Keypoint_Cluster_Queue = dbscan_classification(octree_resolution, eps, minPtsAux, minPts, keypoints_xyzrgb, show);

    std::cout << "-----back in main-----" << std::endl;
    std::cout << "Size of queue = " << Keypoint_Cluster_Queue.size() << endl;
    if(show){
        visualize_clusters(Keypoint_Cluster_Queue, src);
        //visualize_clusters(Keypoint_Cluster_Queue, keypoints_xyzrgb);
    }
}



queue<KeypointCluster> get_cluster_descriptors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &src, pcl::PointCloud<pcl::PointXYZ>::Ptr & keypoints_src_visualize_temp,float min_scale, int nr_octaves, int nr_scales_per_octave,float min_contrast, int show, int octree_resolution, int minPtsAux, int minPts, float eps){

    // ESTIMATING KEY POINTS
    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_src(new pcl::PointCloud<pcl::PointWithScale>);
    cout << "chosen Method is SWIFT" << endl;
    //detect_keypoints(src, keypoints_src);
    detect_keypoints(src, keypoints_src,min_scale,nr_octaves,nr_scales_per_octave,min_contrast);
    cout << "No of SIFT points in the src are " << keypoints_src->points.size() << endl;
    cout << "test 1"  << endl;


    // Copying the pointwithscale to pointxyz so as visualize the cloud
    pcl::copyPointCloud(*keypoints_src, *keypoints_src_visualize_temp);

    if(0 ) {
        pcl::PCDWriter writer;
        pcl::io::savePCDFile("keypoints__visualize_temp.pcd", *keypoints_src_visualize_temp, true);
        pcl::io::savePCDFile("src_cloud.pcd", *src, true);
    }

    //-------------------------------------------------------------------------------------------
    pcl::PointCloud <pcl::PointXYZRGB>::Ptr keypoints_xyzrgb(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::copyPointCloud(*keypoints_src, *keypoints_xyzrgb);


    //new implementaion part
    //do some filtering on the cloud to remove outliers
    if(1){
        // Create the filtering object for RadiusOutlierRemoval
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
        //std::cerr << "setRadiusSearch: " <<test_double1<< std::endl;
        //std::cerr << "setMinNeighborsInRadius: " <<test_double2<< std::endl;
        outrem.setRadiusSearch(5.);//good 5 and r = 3//0.8
        outrem.setMinNeighborsInRadius (4);//2
        std::cout << "keypoints_src_visualize_temp after StatisticalOutlierRemoval: " <<keypoints_xyzrgb->size()<< std::endl;
        outrem.setInputCloud(keypoints_xyzrgb);
        outrem.filter (*keypoints_xyzrgb);
        std::cout << "keypoints_src_visualize_temp after RadiusOutlierRemoval: " <<keypoints_xyzrgb->size()<< std::endl;
    }

    queue<KeypointCluster> Keypoint_Cluster_Queue;
    Keypoint_Cluster_Queue = dbscan_classification(octree_resolution, eps, minPtsAux, minPts, keypoints_xyzrgb, show);

    std::cout << "-----back in main-----" << std::endl;
    std::cout << "Size of queue = " << Keypoint_Cluster_Queue.size() << endl;
    /*
    if(1){
        int tmp_size=Keypoint_Cluster_Queue.size();
        for(int counter=0;counter<tmp_size;counter++){
            KeypointCluster tmp_cluster = Keypoint_Cluster_Queue.front();
            // ESTIMATING PFH FEATURE DESCRIPTORS AT KEYPOINTS AFTER COMPUTING NORMALS
            pcl::PointCloud <pcl::Normal>::Ptr src_normals(new pcl::PointCloud<pcl::Normal>);
            compute_normals(src, src_normals);
            // PFHRGB Estimation
            pcl::PointCloud <pcl::PFHRGBSignature250>::Ptr fpfhs_src_rgb(new pcl::PointCloud<pcl::PFHRGBSignature250>);

            tmp_cluster.fpfhs_rgb_prt =&tmp_cluster.fpfhs_rgb;

            //compute_PFHRGB_features(src, src_normals, keypoints_src, fpfhs_src_rgb);
            compute_PFHRGB_features(src, src_normals, keypoints_src, Cluster1.fpfhs_rgb_prt);
            cout << "End of compute_FPFH_RGB_features! " << endl;


            //first to last of queue
            Keypoint_Cluster_Queue.pop();
            Keypoint_Cluster_Queue.push(tmp_cluster);
        }
    }
     */
    if(show){
        visualize_clusters(Keypoint_Cluster_Queue, src);
    }
    return Keypoint_Cluster_Queue;
}

pcl::visualization::PCLVisualizer::Ptr rgbVis_keypoints (SubmapsVec& submaps_set, int num, bool jet_flag, double jet_stacking_threshold, float min_scale, int nr_octaves, int nr_scales_per_octave,float min_contrast, int show, int octree_resolution, int minPtsAux, int minPts, float eps){
    int vp1_;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    //float black = 0.0;  // Black
    //float white = 1.0 - black;
    viewer->createViewPort (0.0, 0.0, 1.0, 1.0, vp1_);

    unsigned int i = 0;
    PointCloudRGB::Ptr submap_ptr (new PointCloudRGB);
    for(SubmapObj& submap: submaps_set){
        // Find max and min depth in map
        PointT min, max;
        pcl::getMinMax3D(submap.submap_pcl_, min, max);
        std::cout << "Max " << max.getArray3fMap().transpose() << std::endl;
        std::cout << "Min " << min.getArray3fMap().transpose() << std::endl;
        // Normalize and give colors based on z
        for(PointT& pointt: submap.submap_pcl_.points){
            pcl::PointXYZRGB pointrgb;
            pointrgb.x = pointt.x;
            pointrgb.y = pointt.y;
            pointrgb.z = pointt.z;

            std::tuple<uint8_t, uint8_t, uint8_t> colors_rgb;
            if(jet_flag==true) {
                colors_rgb = stacked_jet(pointt.z, jet_stacking_threshold);
            }else{
                colors_rgb = jet((pointt.z - min.z)/(max.z - min.z));
            }

            std::uint32_t rgb = (static_cast<std::uint32_t>(std::get<0>(colors_rgb)) << 16 |
                                 static_cast<std::uint32_t>(std::get<1>(colors_rgb)) << 8 |
                                 static_cast<std::uint32_t>(std::get<2>(colors_rgb)));
            pointrgb.rgb = *reinterpret_cast<float*>(&rgb);
            submap_ptr->points.push_back(pointrgb);
        }
        //--------------------------------------------
        //parts from the following code can be found in the PCL tutorial berkley -> implementation of an FAST keypoint detection

        // Create some new point clouds to hold our data
        pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints (new pcl::PointCloud<pcl::PointWithScale>);

        // Visualize the point cloud and its coordinate system
        std::cout << submap_ptr->points.size() << std::endl;
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_h(submap_ptr);
        viewer->addPointCloud(submap_ptr, rgb_h, "gt_cloud_" + std::to_string(i), vp1_);
        viewer->addCoordinateSystem(3.0, submap.submap_tf_, "gt_cloud_" + std::to_string(i), vp1_);

        //keypoint detection
        //pcl::compute_featuresPointCloud<pcl::PointXYZ>::Ptr keypoints_src_visualize_temp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_src_visualize_temp(new pcl::PointCloud<pcl::PointXYZ>);

        //compute_features( submap_ptr,  keypoints_src_visualize_temp, min_scale,nr_octaves,nr_scales_per_octave,min_contrast,show,octree_resolution, minPtsAux, minPts,eps);
        submap.submap_Keypoint_Cluster_Queue=get_cluster_descriptors( submap_ptr,  keypoints_src_visualize_temp, min_scale,nr_octaves,nr_scales_per_octave,min_contrast,show,octree_resolution, minPtsAux, minPts,eps);


        std::cout << "Num of keypoints: " << keypoints_src_visualize_temp->size() << std::endl;
        viewer->addPointCloud<pcl::PointXYZ>(keypoints_src_visualize_temp, "keypoints_src_corresp_viewer"+ std::to_string(i));
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints_src_corresp_viewer"+ std::to_string(i));


        /*
        //--------------------------------------------
        detect_keypoints (submap_ptr, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints);
        // Visualize the point cloud and its keypoints
        //visualize_keypoints (points, keypoints);
        // Draw each keypoint as a sphere
        for (size_t i = 0; i < keypoints->size (); ++i)
        {
            // Get the point data
            const pcl::PointWithScale & p = keypoints->points[i];

            // Pick the radius of the sphere *
            float r = 2 * p.scale;
            // * Note: the scale is given as the standard deviation of a Gaussian blur, so a
            //   radius of 2*p.scale is a good illustration of the extent of the keypoint

            // Generate a unique string for each sphere
            std::stringstream ss ("keypoint");
            ss << i;

            // Add a sphere at the keypoint
            viewer->addSphere (p, 8.0, 1.0, 0.0, 0.0, ss.str ());
        }
        */
        i++;
    }
    int counter=1;
    for(SubmapObj& submap: submaps_set){
        std::cout << "Submap  " << counter << " contains " << submap.submap_Keypoint_Cluster_Queue.size() << " clusters of keypoints" << std::endl;
        std::cout << "Submap  " << counter << " contains " << submap.submap_pcl_.size() << " points" << std::endl;
        counter++;
    }

    viewer->setBackgroundColor (0, 0, 0, vp1_);
    return (viewer);
}






int main(int argc, char** argv){

    // Inputs
    std::string folder_str, path_str, output_str, simulation;
    int submap_num=1;
    //for dbscan
    int show, octree_resolution, minPtsAux, minPts;
    float eps;

    float test_float=0;
    double test_double1=0, test_double2=0, test_double3=0,test_double4=0;
    float min_scale=0.3, min_contrast=0.25;
    int nr_octaves=4, nr_scales_per_octave=5;
    bool do_keypoints=false, jet_flag=false, filter_flag=true;
    double jet_stacking_threshold;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
        ("help", "Print help")
        ("simulation", "Simulation data from Gazebo", cxxopts::value(simulation))
        ("bathy_survey", "Input MBES pings in cereal file if simulation = no. If in simulation"
                          "input path to map_small folder", cxxopts::value(path_str))
        ("n,submap_num","Number of submaps fromed from the input data", cxxopts::value<int>(submap_num))
        ("s,show", "show visualization", cxxopts::value<int>(show)->default_value("0"))
        ("test_float", "An test float", cxxopts::value<float>(test_float))
        ("t_d1", "An test double", cxxopts::value<double>(test_double1))
        ("t_d2", "An test double", cxxopts::value<double>(test_double2))
        ("t_d3", "An test double", cxxopts::value<double>(test_double3))
        ("t_d4", "An test double", cxxopts::value<double>(test_double4))

        ("octree_resolution", "Param foo", cxxopts::value<int>(octree_resolution)->default_value("100"))
        ("eps", "Param foo", cxxopts::value<float>(eps)->default_value("5.0"))
        ("minPtsAux", "Param foo", cxxopts::value<int>(minPtsAux)->default_value("6"))
        ("minPts", "Param foo", cxxopts::value<int>(minPts)->default_value("2"))

        ("min_scale", "An test double", cxxopts::value<float>(min_scale))
        ("nr_octaves", "An test double", cxxopts::value<int>(nr_octaves))
        ("nr_scales_per_octave", "An test double", cxxopts::value<int>(nr_scales_per_octave))
        ("min_contrast", "An test double", cxxopts::value<float>(min_contrast))

        ("j,jet", "apply jet function to color", cxxopts::value<bool>(jet_flag))
        ("f,filter", "set to false -> no filtering", cxxopts::value<bool>(filter_flag))
        ("i,staked_height", "the height for the stacked color", cxxopts::value<double>(jet_stacking_threshold)->default_value("30.0"))
        ("k,do_keypoints", "scan for keypoints", cxxopts::value<bool>(do_keypoints));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }
    if(output_str.empty()){
        output_str = "output_cereal.cereal";
    }
    if (result.count("i") ){
        std::cout << "Stacked height threshold: " << jet_stacking_threshold << std::endl;
    }
    boost::filesystem::path output_path(output_str);
    string outFilename = "graph_corrupted.g2o";   // G2O output file


    ///add the option to split the data flexible into submaps
    if (result.count("submap_num")){
        if(submap_num<1){
            submap_num=1;
        }
        std::cout << "submap_num = " << result["submap_num"].as<int>() << std::endl;
    }
    if (result.count("test_float")){

        std::cout << "test_float = " << result["test_float"].as<float>() << std::endl;
    }


    // Parse submaps from cereal file
    boost::filesystem::path submaps_path(path_str);
    std::cout << "Input data " << submaps_path << std::endl;

    SubmapsVec submaps_gt; /// SubmapObj is the main class you're going to be working with, familiarize yourself with it
    if(simulation == "yes"){
        submaps_gt = readSubmapsInDir(submaps_path.string());
    }
    else{
        std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(submaps_path);
        std::cout << "Number of pings in survey " << std_pings.size() << std::endl;
        {
            // Parse MBES pings
            SubmapsVec traj_pings = parsePingsAUVlib(std_pings);
            /// Number of pings per submap. if = traj_pings.size()-1
            /// the whole survey is treated as one map.
            /// This is a naive submap construction method. You'll come up with something better :)

            //int submap_size = traj_pings.size()-1;        ///original
            int submap_size = (traj_pings.size())/(submap_num+1);


            /// Construct submaps aggregating pings. The pings are given wrt the map frame, not the AUV frame.
            /// Keep it in mind although it doesn't make a diff for you for now
            submaps_gt = createSubmaps(traj_pings, submap_size);

            // Filtering of submaps
            PointCloudT::Ptr cloud_ptr (new PointCloudT);
            pcl::UniformSampling<PointT> us_filter;
            us_filter.setInputCloud (cloud_ptr);
            std::cerr << "UniformSamplingRadius: " <<test_double4<< std::endl;
            us_filter.setRadiusSearch(test_double4); //1  // See radius of filtering (see Uniform sampling on PCL)

            // Create the filtering object for StatisticalOutlierRemoval
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setMeanK (50);//50
            std::cerr << "setStddevMulThresh: " <<test_double3<< std::endl;
            sor.setStddevMulThresh (test_double3);//1.0

            // Create the filtering object for RadiusOutlierRemoval
            pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
            std::cerr << "setRadiusSearch: " <<test_double1<< std::endl;
            std::cerr << "setMinNeighborsInRadius: " <<test_double2<< std::endl;
            outrem.setRadiusSearch(test_double1);//0.8
            outrem.setMinNeighborsInRadius (test_double2);//2


            for(SubmapObj& submap_i: submaps_gt){
                *cloud_ptr = submap_i.submap_pcl_;
                std::cerr << "Cloud before filtering: " <<cloud_ptr->size()<< std::endl;
                us_filter.setInputCloud(cloud_ptr);
                us_filter.filter(*cloud_ptr);
                std::cerr << "Cloud after Uniform sampling: " <<cloud_ptr->size()<< std::endl;

                if(filter_flag==true){
                    sor.setInputCloud (cloud_ptr);
                    sor.filter (*cloud_ptr);
                    std::cerr << "Cloud after StatisticalOutlierRemoval: " <<cloud_ptr->size()<< std::endl;
                    outrem.setInputCloud(cloud_ptr);
                    outrem.filter (*cloud_ptr);
                    std::cerr << "Cloud after RadiusOutlierRemoval: " <<cloud_ptr->size()<< std::endl;
                }
                submap_i.submap_pcl_ = *cloud_ptr;
            }
        }
    }

    /// If you want to visualize the survey as a single map (submap_size = traj_pings.size()-1)
    /// with depth colours, FULLMAP==1
    /// If you're working with submaps and would like to see them with different colours, SUBMAPS==1
    /// Don't visualize the submaps with depth colours since these are computed for each submap and they
    /// won't make sense

    // Visualization
#if FULLMAP == 1
    pcl::visualization::PCLVisualizer::Ptr viewer;
    //viewer = rgbVis(submaps_gt, 1);
    if(do_keypoints){
        //if(do_keypoints){
            std::cout << "do keypoints is on";
            viewer = rgbVis_keypoints(submaps_gt, 1,jet_flag, jet_stacking_threshold, min_scale,nr_octaves, nr_scales_per_octave,min_contrast, show,octree_resolution, minPtsAux, minPts,eps);
    }else{
        viewer = rgbVis(submaps_gt, 1, jet_flag, jet_stacking_threshold);
    }

    while(!viewer->wasStopped ()){
        viewer->spinOnce ();
    }
    viewer->resetStoppedFlag();
#endif

#if SUBMAPS == 1
    PCLVisualizer viewer ("Submaps viewer");
    SubmapsVisualizer* visualizer = new SubmapsVisualizer(viewer);
    visualizer->setVisualizer(submaps_gt, 1);

    while(!viewer.wasStopped ()){
        viewer.spinOnce ();
    }
    viewer.resetStoppedFlag();
#endif



    return 0;
}
