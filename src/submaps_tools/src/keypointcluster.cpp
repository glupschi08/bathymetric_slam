//
// Created by florian on 09/07/20.
//

#include <stdlib.h>
//#include "keypointcluster.h"

#include "submaps_tools/keypointcluster.h"
//#include <Eigen/Core>
//#include <pcl/visualization/pcl_plotter.h>

// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
#define feature_radius 5.25//3.25  //0.25
#define RANSAC_Inlier_Threshold 3.//1.5 //0.1
#define RANSAC_Iterations 5000

float colors[] = {
        255, 0,   0,   // red 		1
        0,   255, 0,   // green		2
        0,   0,   255, // blue		3
        255, 255, 0,   // yellow		4
        0,   255, 255, // light blue	5
        255, 0,   255, // magenta     6
        255, 255, 255, // white		7
        255, 128, 0,   // orange		8
        255, 153, 255, // pink		9
        51,  153, 255, //			10
        153, 102, 51,  //			11
        128, 51,  153, //			12
        153, 153, 51,  //			13
        163, 38,  51,  //			14
        204, 153, 102, //		15
        204, 224, 255, //		16
        128, 179, 255, //		17
        206, 255, 0,   //			18
        255, 204, 204, //			19
        204, 255, 153, //			20

};


void KeypointCluster::set_cloud(int ClusterID_tmp, PointCloudRGBT& key_cloud_tmp){
    key_cloud=key_cloud_tmp;
    ClusterID=ClusterID_tmp;
};


void KeypointCluster::set_values ( int y_size,PointT Centroid_p,Eigen::Matrix3f e_vecs,Eigen::Vector3f e_vals, float a, float b, float c, float d, float e, float f) {
    Centroid=Centroid_p;
    ClusterSize=y_size;
    Eigenvectors=e_vecs;
    Eigenvalues=e_vals;
    minX = a;
    minY = b;
    minZ = c;
    maxY = d;
    maxX = e;
    maxZ = f;
}




void visualize_clusters(queue<KeypointCluster> Keypoint_Cluster_Queue, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){

    //vtkObject::GlobalWarningDisplayOff(); // Disable vtk render warning

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("DBSCAN CLUSTERS"));

    int PORT1 = 0;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, PORT1);
    viewer->setBackgroundColor(0, 0, 0, PORT1);
    viewer->addText("ORIGINAL", 10, 10, "PORT1", PORT1);

    int PORT2 = 0;
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, PORT2);
    viewer->setBackgroundColor(0, 0, 0, PORT2);
    viewer->addText("CLUSTERS", 10, 10, "PORT2", PORT2);

    int numClust = 0;
    std::random_device seeder;
    std::ranlux48 gen(seeder());
    std::uniform_int_distribution<int> uniform_0_255(0, 255);

    int j = 0;

    viewer->setPosition(0, 0);
    viewer->setBackgroundColor(0.0, 0.0, 0.0, 0.0); // Setting background to a dark
    int tmp_thresh=Keypoint_Cluster_Queue.size();
    for (int counter=0;counter < tmp_thresh;counter++){
        string lineID = to_string(counter);
        pcl::PointXYZ src_idx, tgt_idx;

        KeypointCluster tmp_cluster = Keypoint_Cluster_Queue.front(); //getting the first

        src_idx= tmp_cluster.Centroid;

        float scaller, offset;
        int visu_scale=50, vec_cnt=0;
        scaller =(tmp_cluster.Eigenvalues[0]+tmp_cluster.Eigenvalues[1]+tmp_cluster.Eigenvalues[2]);
        tgt_idx=src_idx;
        offset=visu_scale*(tmp_cluster.Eigenvalues[vec_cnt]/scaller);

        tgt_idx.x=src_idx.x+offset*tmp_cluster.Eigenvectors(vec_cnt,0);
        tgt_idx.y=src_idx.y+offset*tmp_cluster.Eigenvectors(vec_cnt,1);
        tgt_idx.z=src_idx.z+offset*tmp_cluster.Eigenvectors(vec_cnt,2);

        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(src_idx, tgt_idx, 125, 125, 125, "x"+lineID);

        vec_cnt++;
        tgt_idx=src_idx;
        offset=visu_scale*(tmp_cluster.Eigenvalues[vec_cnt]/scaller);

        tgt_idx.x=src_idx.x+offset*tmp_cluster.Eigenvectors(0,vec_cnt);
        tgt_idx.y=src_idx.y+offset*tmp_cluster.Eigenvectors(1,vec_cnt);
        tgt_idx.z=src_idx.z+offset*tmp_cluster.Eigenvectors(2,vec_cnt);
        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(src_idx, tgt_idx, 125, 125, 125, "y"+lineID);

        vec_cnt++;
        tgt_idx=src_idx;
        offset=visu_scale*(tmp_cluster.Eigenvalues[vec_cnt]/scaller);

        tgt_idx.x=src_idx.x+offset*tmp_cluster.Eigenvectors(0,vec_cnt);
        tgt_idx.y=src_idx.y+offset*tmp_cluster.Eigenvectors(1,vec_cnt);
        tgt_idx.z=src_idx.z+offset*tmp_cluster.Eigenvectors(2,vec_cnt);
        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(src_idx, tgt_idx, 125, 125, 125, "z"+lineID);

        //first to last of queue
        Keypoint_Cluster_Queue.pop();
        Keypoint_Cluster_Queue.push(tmp_cluster);
    }

    cout << "after the dequeuing Size of queue = " << Keypoint_Cluster_Queue.size() << endl;
    for (int counter=0;counter < tmp_thresh;counter++){
        KeypointCluster tmp_cluster = Keypoint_Cluster_Queue.front();
        cout << "Size of cluster = " << tmp_cluster.key_cloud.size() << endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
        uint8_t r;
        uint8_t g;
        uint8_t b;

        if (j < 60) {
            r = (uint8_t)colors[j];
            g = (uint8_t)colors[j + 1];
            b = (uint8_t)colors[j + 2];
        } else {
            r = (uint8_t)uniform_0_255(gen);
            g = (uint8_t)uniform_0_255(gen);
            b = (uint8_t)uniform_0_255(gen);
        }

        // Adding different color to each cluster
        for (std::size_t i = 0; i < tmp_cluster.key_cloud.size (); ++i){
            pcl::PointXYZRGB point;
            point.x = tmp_cluster.key_cloud[i].x;
            point.y = tmp_cluster.key_cloud[i].y;
            point.z = tmp_cluster.key_cloud[i].z;
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            point.rgb = *reinterpret_cast<float *>(&rgb);
            cluster_rgb->points.push_back(point);
        }

        j += 3;

        std::string nameId = "cluster_";
        nameId += std::to_string(numClust);

        viewer->addPointCloud(cluster_rgb, nameId.c_str(), PORT2);
        nameId += "_";
        nameId += std::to_string(numClust);
        viewer->addPointCloud(cluster_rgb, nameId.c_str(), PORT1);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5,  nameId.c_str());
        numClust += 1;

        //first to last of queue
        Keypoint_Cluster_Queue.pop();
        Keypoint_Cluster_Queue.push(tmp_cluster);
    }

    double scale = 1;
    viewer->addCoordinateSystem(scale);
    pcl::PointXYZ p11, p22, p33;
    p11.getArray3fMap() << 1, 0, 0;
    p22.getArray3fMap() << 0, 1, 0;
    p33.getArray3fMap() << 0, 0.1, 1;

    viewer->addText3D("x", p11, 0.2, 1, 0, 0, "x_");
    viewer->addText3D("y", p22, 0.2, 0, 1, 0, "y_");
    viewer->addText3D("z", p33, 0.2, 0, 0, 1, "z_");

    if (cloud->points[0].r <= 0 and cloud->points[0].g <= 0 and cloud->points[0].b <= 0) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_handler(cloud, 255, 255, 0);
        viewer->addPointCloud(cloud, color_handler, "Original_Cloud", PORT1);
    } else {
        viewer->addPointCloud(cloud, "Original_Cloud", PORT1);
    }

    viewer->initCameraParameters();
    viewer->resetCamera();

    pcl::console::print_info("\n- clusters: ");
    pcl::console::print_value("%d", numClust);
    pcl::console::print_info("\npress [q] to exit!\n");
/*
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
    */
}

/*
void findCorrespondences_PFHRGB(const pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr &fpfhs_src,
                                const pcl::PointCloud<pcl::PFHRGBSignature250>::Ptr &fpfhs_tgt,
                                pcl::Correspondences &all_correspondences) {
    pcl::registration::CorrespondenceEstimation<pcl::PFHRGBSignature250, pcl::PFHRGBSignature250> est;
    est.setInputSource(fpfhs_src);
    est.setInputTarget(fpfhs_tgt);
    est.determineReciprocalCorrespondences(all_correspondences);
}

void rejectBadCorrespondences(const pcl::CorrespondencesPtr &all_correspondences,
                              const pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_src,
                              const pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_tgt,
                              pcl::Correspondences &remaining_correspondences){
    // copy only XYZRGB data of keypoints for use in estimating features
    pcl::PointCloud <pcl::PointXYZRGB>::Ptr keypoints_src_xyzrgb(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::PointCloud <pcl::PointXYZRGB>::Ptr keypoints_tgt_xyzrgb(new pcl::PointCloud <pcl::PointXYZRGB>);
    pcl::copyPointCloud(*keypoints_src, *keypoints_src_xyzrgb);
    pcl::copyPointCloud(*keypoints_tgt, *keypoints_tgt_xyzrgb);

    // RandomSampleConsensus bad correspondence rejector
    pcl::registration::CorrespondenceRejectorSampleConsensus <pcl::PointXYZRGB> correspondence_rejector;
    correspondence_rejector.setInputSource (keypoints_src_xyzrgb);
    correspondence_rejector.setInputTarget (keypoints_tgt_xyzrgb);
    correspondence_rejector.setInlierThreshold(RANSAC_Inlier_Threshold);
    correspondence_rejector.setMaximumIterations(RANSAC_Iterations);
    correspondence_rejector.setRefineModel(true);//false
    correspondence_rejector.setInputCorrespondences(all_correspondences);
    correspondence_rejector.getCorrespondences(remaining_correspondences);
}
*/

/*
queue<KeypointCluster> dbscan_classification(int octreeResolution, float eps, int minPtsAux, int minPts, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int show){
//void init(int filter_flag, int octreeResolution, float eps, int minPtsAux, int minPts, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int show){

    queue<KeypointCluster> Keypoint_Cluster_Queue;
    std::vector<htr::Point3D> groupA;
    dbScanSpace::dbscan dbscan;


    // K nearest neighbor search
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_xyz);
    pcl::PointXYZ searchPoint;
    // ... populate the cloud and the search point
    // create a kd-tree instance
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    // assign a point cloud - this builds the tree
    kdtree.setInputCloud(cloud_xyz);
    // pre-allocate the neighbor index and
    // distance vectors
    int K = 10;
    std::vector<int> pointsIdx(K);
    std::vector<float> pointsSquaredDist(K);
    // K nearest neighbor search
    kdtree.nearestKSearch(searchPoint, K, pointsIdx, pointsSquaredDist);

    std::cout << "K nearest neighbor search at (" << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z
              << ") with K=" << K << std::endl;

    if (kdtree.nearestKSearch(searchPoint, K, pointsIdx, pointsSquaredDist) > 0) {
        for (std::size_t i = 0; i < pointsIdx.size(); ++i)
            std::cout << "    " << cloud->points[pointsIdx[i]].x << " " << cloud->points[pointsIdx[i]].y << " "
                      << cloud->points[pointsIdx[i]].z << " (squared distance: " << pointsSquaredDist[i] << ")" << std::endl;
    }

    std::vector<double> doubleVec(pointsSquaredDist.begin(), pointsSquaredDist.end());
    double min_val = *std::min_element(doubleVec.begin(), doubleVec.end());
    double max_val = *std::max_element(doubleVec.begin(), doubleVec.end());
    std::vector<double> doubleVec_normalized;

    for (double x : doubleVec) {
        double norm = ((x - min_val) / (max_val - min_val));
        doubleVec_normalized.push_back(norm);
    }

    std::partial_sort(doubleVec_normalized.begin(), doubleVec_normalized.begin() + 2, doubleVec_normalized.end());
    std::cout << "Sorted squared distances (normalized): \n";

    for (auto x : doubleVec_normalized)
        std::cout << x << std::endl;

    std::vector<double> doubleVec_X;
    double cont_x = 0;

    for (int x = 0; x < 300; x++) {
        doubleVec_X.push_back(cont_x);
        cont_x += 500;
    }

    // defining a plotter
    pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter;
    // adding the polynomial func1 to the plotter with [-10, 10] as the range in X axis and "y = x^2" as title
    plotter->addPlotData(doubleVec_normalized, doubleVec_X, "k square distance", vtkChart::LINE, std::vector<char>());
    plotter->spinOnce(300);

    std::cerr << "octreeResolution: " <<octreeResolution<< std::endl;
    std::cerr << "eps: " <<eps<< std::endl;
    std::cerr << "minPtsAux_: " <<minPtsAux<< std::endl;
    std::cerr << "minPts: " <<minPts<< std::endl;
    //----------------------------------------------------------------
    dbscan.init(groupA, cloud, octreeResolution, eps, minPtsAux, minPts);
    //----------------------------------------------------------------

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    // dbscan.generateClusters();
    dbscan.generateClusters_fast();

    ofstream fout;
    int cont = 0;
    if (dbscan.getClusters().size() <= 0) {

        pcl::console::print_error("\nCould not generated clusters, bad parameters\n");
        std::exit(-1);
    }
    std::cout << "num of clusters:"<<  dbscan.getClusters().size()  <<"\n";

    int cluster_cnt =1;
    for (auto &cluster : dbscan.getClusters()) {
        cluster_cnt++;
        std::string str1 = "../";
        str1 += "/cloud_cluster_";
        str1 += std::to_string(cont);
        str1 += ".pcd";

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_pcd(new pcl::PointCloud<pcl::PointXYZRGB>());

        // std::cout << "\nPrinting clusters..." << std::endl;
        std::cout << "cluster " << cluster_cnt << ":" << cluster.clusterPoints.size() << std::endl;
        std::cout << "cluster clustersCentroids" << cluster_cnt << ":" << cluster.centroid << std::endl;
        //std::cout << "cluster clustersCentroids" << cluster_cnt << "x :" << cluster.centroid.x << std::endl;
        pcl::PointXYZ centroit_point;
        centroit_point.x=cluster.centroid.x;
        centroit_point.y=cluster.centroid.y;
        centroit_point.z=cluster.centroid.z;
        //std::cout << "cluster clustersCentroids" << cluster_cnt << "point.x :" << point.x << std::endl;
        for (auto &point : cluster.clusterPoints) {
            pcl::PointXYZRGB pt;
            pt.x = point.x;
            pt.y = point.y;
            pt.z = point.z;
            pt.r = point.r;
            pt.g = point.g;
            pt.b = point.b;
            cloud_cluster_pcd->points.push_back(pt);
        }

        pcl::PointXYZRGB minPt, maxPt;
        pcl::getMinMax3D (*cloud_cluster_pcd, minPt, maxPt);
        std::cout << "Max x: " << maxPt.x << std::endl;
        std::cout << "Max y: " << maxPt.y << std::endl;
        std::cout << "Max z: " << maxPt.z << std::endl;
        std::cout << "Min x: " << minPt.x << std::endl;
        std::cout << "Min y: " << minPt.y << std::endl;
        std::cout << "Min z: " << minPt.z << std::endl;

        //Source: http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html
        // Compute principal directions
        Eigen::Vector4f pcaCentroid;
        pcl::compute3DCentroid(*cloud_cluster_pcd, pcaCentroid);
        Eigen::Matrix3f covariance;
        computeCovarianceMatrixNormalized(*cloud_cluster_pcd, pcaCentroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
        eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));  /// This line is necessary for proper orientation in some cases. The numbers come out the same without it, but
        // Note that getting the eigenvectors can also be obtained via the PCL PCA interface with something like:
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PCA<pcl::PointXYZRGB> pca;
        pca.setInputCloud(cloud_cluster_pcd);
        pca.project(*cloud_cluster_pcd, *cloudPCAprojection);
        std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;
        std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;

        //save the information in the ClusterDescriptor
        KeypointCluster Cluster1;


        Cluster1.set_values(cloud_cluster_pcd->size(),centroit_point,pca.getEigenVectors(),pca.getEigenValues(),minPt.x, minPt.y, minPt.z,maxPt.x, maxPt.y, maxPt.z);
        //std::cout << "Cluster1 Eigenvalues: " << Cluster1.Eigenvalues << std::endl;
        //Cluster1.set_values(1,cloud_cluster_pcd->size(),minPt.x, minPt.y, minPt.z,maxPt.x, maxPt.y, maxPt.z);
        std::cout << "Cluster1 Eigenvalues: " << Cluster1.Eigenvalues << std::endl;
        Cluster1.set_cloud(cluster_cnt, *cloud_cluster_pcd);
        Keypoint_Cluster_Queue.push( Cluster1 );
        pcl::io::savePCDFileBinary(str1.c_str(), *cloud_cluster_pcd);
        cont += 1;
    }
    cout << "Size of queue = " << Keypoint_Cluster_Queue.size() << endl;

    //-------------------------------------------------------------------//
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    pcl::console::print_info("\n- elapsed time: ");
    pcl::console::print_value("%d", elapsed_seconds.count());

    return Keypoint_Cluster_Queue;
}
*/