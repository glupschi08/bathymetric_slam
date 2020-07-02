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
//----------------

#define SUBMAPS 0
#define FULLMAP 1

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
    double r, g, b, val;
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

pcl::visualization::PCLVisualizer::Ptr rgbVis (SubmapsVec& submaps_set, int num){
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
            colors_rgb = jet((pointt.z - min.z)/(max.z - min.z));
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

    viewer->setBackgroundColor (white, white, white, vp1_);
    viewer->setBackgroundColor (0,0,0);
    //viewer->setBackgroundColor (black, black, black, vp1_);

    return (viewer);
}

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




void visualize_keypoints (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points, const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints){
    // Add the points to the vizualizer
    pcl::visualization::PCLVisualizer viz;
    viz.addPointCloud (points, "points");

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
        viz.addSphere (p, 2*p.scale, 1.0, 0.0, 0.0, ss.str ());
    }

    // Give control over to the visualizer
    viz.spin ();
}

pcl::visualization::PCLVisualizer::Ptr rgbVis_keypoints (SubmapsVec& submaps_set, int num){
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
            colors_rgb = jet((pointt.z - min.z)/(max.z - min.z));
            std::uint32_t rgb = (static_cast<std::uint32_t>(std::get<0>(colors_rgb)) << 16 |
                                 static_cast<std::uint32_t>(std::get<1>(colors_rgb)) << 8 |
                                 static_cast<std::uint32_t>(std::get<2>(colors_rgb)));
            pointrgb.rgb = *reinterpret_cast<float*>(&rgb);
            submap_ptr->points.push_back(pointrgb);
        }


        //--------------------------------------------
        //this part is from the PCL tutorial berkley -> implementation of an FAST keypoint detection

        // Create some new point clouds to hold our data
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints (new pcl::PointCloud<pcl::PointWithScale>);

        // Compute keypoints
        const float min_scale = 0.01;
        const int nr_octaves = 3;
        const int nr_octaves_per_scale = 3;
        const float min_contrast = 10.0;
        //detect_keypoints (points, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints);

        // Visualize the point cloud and its keypoints
        //visualize_keypoints (points, keypoints);
        //---------------------------------


        std::cout << submap_ptr->points.size() << std::endl;
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_h(submap_ptr);
        viewer->addPointCloud(submap_ptr, rgb_h, "gt_cloud_" + std::to_string(i), vp1_);
        viewer->addCoordinateSystem(3.0, submap.submap_tf_, "gt_cloud_" + std::to_string(i), vp1_);


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
            //viewer->addSphere (p, 2*p.scale, 1.0, 0.0, 0.0, ss.str ());
            viewer->addSphere (p, 8.0, 1.0, 0.0, 0.0, ss.str ());
        }
        //--------------------------------------------
        i++;
    }

    //viewer->setBackgroundColor (white, white, white, vp1_);
    //viewer->setBackgroundColor (0,0,0);
    viewer->setBackgroundColor (black, black, black, vp1_);

    return (viewer);
}


int main(int argc, char** argv){

    // Inputs
    std::string folder_str, path_str, output_str, simulation;
    int submap_num=5;
    float test_float=0;
    bool do_keypoints=false;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
        ("help", "Print help")
        ("simulation", "Simulation data from Gazebo", cxxopts::value(simulation))
        ("bathy_survey", "Input MBES pings in cereal file if simulation = no. If in simulation"
                          "input path to map_small folder", cxxopts::value(path_str))
        ("submap_num","Number of submaps fromed from the input data", cxxopts::value<int>(submap_num))
        ("test_float", "An test float", cxxopts::value<float>(test_float))
        ("k,do_keypoints", "scan for keypoints", cxxopts::value<bool>(do_keypoints));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }
    if(output_str.empty()){
        output_str = "output_cereal.cereal";
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
            us_filter.setRadiusSearch(1);   // See radius of filtering (see Uniform sampling on PCL)
            for(SubmapObj& submap_i: submaps_gt){
                *cloud_ptr = submap_i.submap_pcl_;
                us_filter.setInputCloud(cloud_ptr);
                us_filter.filter(*cloud_ptr);
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
            viewer = rgbVis_keypoints(submaps_gt, 1);
    }else{
        viewer = rgbVis(submaps_gt, 1);
    }


    //----------------------test area for viso
    //viewer_.addArrow(PointT(0,0,0, PointT(to_ps[0],to_ps[1],to_ps[2]),dr_color[0], dr_color[1], dr_color[2], false, "gt_dr_edge_" + std::to_string(j), vp1_);
    /*
    pcl::ModelCoefficients coeffs;
    coeffs.values.push_back(0.3);
    coeffs.values.push_back(0.3);
    coeffs.values.push_back(0.0);
    coeffs.values.push_back(0.0);
    coeffs.values.push_back(1.0);
    coeffs.values.push_back(0.0);
    coeffs.values.push_back(5.0);
    viewer->addCone (coeffs, "cone");
    std::cout << "cone test";
    coeffs.values.clear ();
    coeffs.values.push_back(-10.);                 //x offset
    coeffs.values.push_back(0.0);                   //Y offset
    coeffs.values.push_back(4.0);                  //z offset
    coeffs.values.push_back(0.0);                   //change angle and size
    coeffs.values.push_back(25.0);
    coeffs.values.push_back(2.0);
    coeffs.values.push_back(10.0);
    viewer->addCone (coeffs, "cone2");
    std::cout << "cone test";
     */
    //viewer->addSphere ((), 0.2, 0.5, 0.5, 0.0, "sphere");
    //-----------------------------end of test area


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
