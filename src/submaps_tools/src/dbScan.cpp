//#include "include/dbScan.h"
#include "submaps_tools/dbScan.h"

namespace dbScanSpace {
dbscan::dbscan() {
  octreeGenIn = new htr::OctreeGenerator();
  octreeGen = new htr::OctreeGenerator();
}

dbscan::~dbscan() {
  delete octreeGenIn;
  delete octreeGen;
}

/// Initializes the point cloud from a file, and the octree.
///@param[in] filename          Location of the file that has the data.
///@param[in] octreeResolution_ Resolution with which the octree is initialized.
///@param[in] eps_              The search radius for the octree.
///@param[in] minPtsAux_        Minimum points for the initial clusters.
///@param[in] minPts_           Minimum points for the final clusters.

void dbscan::generateClusters_one_step() {
  clusters.clear();
  clustersAux.clear();
  clustersCentroids.clear();

  // A first set of clusters is generated. This first set has a large number of small clusters.
  DBSCAN_Octree_fast_one_step(octreeGenIn, eps, minPts);

  //// The clusters centroids are calculated and used to generate a second octree.
  // for (dbScanSpace::cluster cluster : clustersAux)
  //	clustersCentroids.push_back(cluster.centroid);

  // octreeGen->initCloudFromVector<pcl::mod_pointXYZ>(clustersCentroids);
  // octreeGen->initOctree(octreeResolution);

  //// Using the second octree and the centroids of the clusters, a new set of clusters is
  /// generated.
  //// These are the final clusters.
  // DBSCAN_Octree_merge(octreeGen, 2 * eps, minPts);

  for (int i = 0; i < clusters.size(); i++)
    clusters[i].toPoint3D();
}

void CloudToVector(const std::vector<pcl::mod_pointXYZ> &inPointVector,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr &outPointCloud) {
  for (const pcl::mod_pointXYZ &point : inPointVector) {
    outPointCloud->points.push_back(point);
  }
}

/// Generates the clusters from the loaded data.
void dbscan::generateClusters() {
  clusters.clear();
  clustersAux.clear();
  clustersCentroids.clear();

  // A first set of clusters is generated. This first set has a large number of small clusters.
  DBSCAN_Octree(octreeGenIn, eps, minPtsAux);

  // The clusters centroids are calculated and used to generate a second octree.
  for (dbScanSpace::cluster cluster : clustersAux)
    clustersCentroids.push_back(cluster.centroid);

  // octreeGen->initCloudFromVector<pcl::mod_pointXYZ>(clustersCentroids);
  octreeGen->initOctree(octreeResolution);

  // Using the second octree and the centroids of the clusters, a new set of clusters is generated.
  // These are the final clusters.
  DBSCAN_Octree_merge(octreeGen, 2 * eps, minPts);

  for (int i = 0; i < clusters.size(); i++)
    clusters[i].toPoint3D();
}

/// Generates the clusters from the loaded data.
void dbscan::generateClusters_fast() {
  clusters.clear();
  clustersAux.clear();
  clustersCentroids.clear();

  // A first set of clusters is generated. This first set has a large number of small clusters.
  // DBSCAN_Octree_fast2(octreeGenIn, minPtsAux);
  DBSCAN_Octree_fast(octreeGenIn, eps, minPtsAux);

  //        printf("\n Aux clusters size:%d\n\n", clustersAux.size());
  // The clusters centroids are calculated and used to generate a second octree.
  for (dbScanSpace::cluster cluster : clustersAux)
    clustersCentroids.push_back(cluster.centroid);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  CloudToVector(clustersCentroids, new_cloud);
  octreeGen->initCloudFromVector<pcl::mod_pointXYZ>(clustersCentroids, new_cloud);
  octreeGen->initOctree(octreeResolution);

  // Using the second octree and the centroids of the clusters, a new set of clusters is generated.
  // These are the final clusters.
  // DBSCAN_Octree_merge(octreeGen, 2*eps, minPts);
  DBSCAN_Octree_merge(octreeGen, 2 * eps, minPts);

  //        printf("\n Clusters size:%d\n\n", clusters.size());
  for (int i = 0; i < clusters.size(); i++)
    clusters[i].toPoint3D();
}

/// Calculates the centroid form a vector of points.
///@param[in] group             Vector that contains the point that will be processed.
void dbscan::calculateCentroid(std::vector<pcl::mod_pointXYZ> group) {
  for (pcl::mod_pointXYZ point : group) {
    centroid.x += point.x;
    centroid.y += point.y;
    centroid.z += point.z;
  }
  centroid.x /= group.size();
  centroid.y /= group.size();
  centroid.z /= group.size();
}

/// Does a radius search for the K nearest neighbors of a point.
///@param[in] octreeGen         The octree to be searched.
///@param[in] searchPoint       The point around which the search will be conducted.
///@param[in] eps_              The search radius for the octree.
///@param[in] retKeys_          Vector that stores the indices of the nearest points.
void dbscan::octreeRegionQuery(htr::OctreeGenerator *octreeGen, pcl::mod_pointXYZ &searchPoint, double eps,
                               std::vector<int> *retKeys) {
  retKeys->clear();
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;

  octreeGen->getOctree()->radiusSearch(searchPoint, eps, *retKeys, pointRadiusSquaredDistance);
}

/// Merges a set of clusters.
///@param[in] octreeGen         The octree to be searched.
///@param[in] eps_              The search radius for the octree.
///@param[in] clustersIn        The clusters that will be merged.
///@param[in] clustersOut       Vector that stores all merged clusters.
void dbscan::DBSCAN_Octree_merge(htr::OctreeGenerator *octreeGen, float eps, int minPts) {
  clusters.clear();
  cluster pointQueue;
  std::vector<pcl::mod_pointXYZ> clusterPoints;

  // The amount of aux clusters
  int noKeys = clustersAux.size();
  std::vector<bool> visited(noKeys, false);

  std::vector<int> noise;
  std::vector<int> neighborPts(noKeys, -1);

  for (int i = 0; i < noKeys; i++) {
    if (!visited[i]) {
      clusterPoints.push_back(clustersAux.at(i).centroid);

      pointQueue.clusterPoints.insert(pointQueue.clusterPoints.end(), clustersAux.at(i).clusterPoints.begin(),
                                      clustersAux.at(i).clusterPoints.end());

      visited[i] = true;

      for (int j = 0; j < clusterPoints.size(); j++) {
        octreeRegionQuery(octreeGen, clusterPoints.at(j), eps, &neighborPts);

        for (int k = 0; k < neighborPts.size(); k++) {
          if (!visited[neighborPts[k]]) {
            visited[neighborPts[k]] = true;

            clusterPoints.push_back(clustersAux.at(neighborPts[k]).centroid);

            pointQueue.clusterPoints.insert(pointQueue.clusterPoints.end(),
                                            clustersAux.at(neighborPts[k]).clusterPoints.begin(),
                                            clustersAux.at(neighborPts[k]).clusterPoints.end());
          }
        }
      }

      if (pointQueue.clusterPoints.size() >= minPts) {
        pointQueue.calculateCentroid();
        clusters.push_back(pointQueue);
        pointQueue.clusterPoints.clear();
      }
    }
  }

  //       clustersAux.clear();
}

/// Generates a set of clusters.
///@param[in] octreeGen         The octree to be searched.
///@param[in] eps_              The search radius for the octree.
///@param[in] clusters          Vector that stores all the generated clusters.
void dbscan::DBSCAN_Octree(htr::OctreeGenerator *octreeGen, float eps, int minPts) {
  clustersAux.clear();
  cluster pointQueue;
  pcl::mod_pointXYZ auxCentroid;

  int noKeys = octreeGen->getCloud()->points.size();
  std::vector<bool> visited(noKeys, false);
  std::vector<int> classification(noKeys, 0);

  std::vector<int> noise;
  std::vector<int> neighborPts;

  for (int i = 0; i < noKeys; i++) {
    if (!visited[i]) {
      pointQueue.clusterPoints.push_back(octreeGen->getCloud()->points.at(i));
      visited[i] = true;

      octreeRegionQuery(octreeGen, pointQueue.clusterPoints.at(0), eps, &neighborPts);

      if (neighborPts.size() < minPtsAux)
        noise.push_back(i);
      else {
        for (int k = 0; k < neighborPts.size(); k++) {
          if (!visited[neighborPts[k]]) {
            visited[neighborPts[k]] = true;
            pcl::mod_pointXYZ auxPoint = octreeGen->getCloud()->points.at(neighborPts[k]);
            pointQueue.clusterPoints.push_back(auxPoint);
          }
        }

        if (pointQueue.clusterPoints.size() >= minPtsAux) {
          pointQueue.calculateCentroid();
          clustersAux.push_back(pointQueue);
          pointQueue.clusterPoints.clear();
        }
      }
    }
  }
}

/// Generates a set of clusters.
///@param[in] octreeGen         The octree to be searched.
///@param[in] eps_              The search radius for the octree.
///@param[in] clusters          Vector that stores all the generated clusters.
void dbscan::DBSCAN_Octree_fast(htr::OctreeGenerator *octreeGen, float eps, int minPts) // O(nlogn)
{
  clustersAux.clear();

  cluster pointQueue;
  pcl::mod_pointXYZ auxCentroid;

  // The number of points
  int noKeys = octreeGen->getCloud()->points.size();
  std::vector<bool> visited(noKeys, false);
  std::vector<int> classification(noKeys, 0);

  std::vector<int> noise;
  std::vector<int> neighborPts;

  for (int i = 0; i < noKeys; i++) // O(n)
  {
    if (!visited[i]) // O(log n)
    {
      pointQueue.clusterPoints.push_back(octreeGen->getCloud()->points.at(i));
      visited[i] = true;

      octreeGen->getOctree()->voxelSearch(pointQueue.clusterPoints.at(0), neighborPts);

      if (neighborPts.size() < minPtsAux)
        noise.push_back(i);
      else {
        for (int k = 0; k < neighborPts.size(); k++) {
          if (!visited[neighborPts[k]]) {
            visited[neighborPts[k]] = true;
            pcl::mod_pointXYZ auxPoint = octreeGen->getCloud()->points.at(neighborPts[k]);
            pointQueue.clusterPoints.push_back(auxPoint);
          }
        }

        if (pointQueue.clusterPoints.size() >= minPtsAux) {
          pointQueue.calculateCentroid();
          clustersAux.push_back(pointQueue);
          pointQueue.clusterPoints.clear();
        }
      }
    }
  }
}

/// Generates a set of clusters.
///@param[in] octreeGen         The octree to be searched.
///@param[in] eps_              The search radius for the octree.
///@param[in] clusters          Vector that stores all the generated clusters.
void dbscan::DBSCAN_Octree_fast_one_step(htr::OctreeGenerator *octreeGen, float eps, int minPts) {
  int noKeys = octreeGen->getCloud()->points.size();
  std::vector<bool> clustered(noKeys, false);
  std::vector<bool> visited(noKeys, false);

  std::vector<int> noise;

  std::vector<int> neighborPts;
  std::vector<int> neighborPts_;

  int c = 0;

  // for each unvisted point P in dataset keypoints
  for (int i = 0; i < noKeys; i++) {
    if (!visited[i]) {
      // Mark P as visited
      visited[i] = true;
      octreeRegionQuery(octreeGen, octreeGen->getCloud()->points.at(i), eps, &neighborPts);

      if (neighborPts.size() < minPts)
        // Mark P as Noise
        noise.push_back(i);
      else {
        clusters.push_back(cluster());

        // expand cluster, add P to cluster c
        clustered[i] = true;
        clusters.at(c).clusterPoints.push_back(octreeGen->getCloud()->points.at(i));

        // for each point P' in neighborPts, Expand cluster
        for (int j = 0; j < neighborPts.size(); j++) {
          //                        if P' is not visited
          if (!visited[neighborPts[j]]) {
            // Mark P' as visited
            visited[neighborPts[j]] = true;
            octreeRegionQuery(octreeGen, octreeGen->getCloud()->points.at(neighborPts[j]), eps, &neighborPts_);
            //
            if (neighborPts_.size() >= minPts)
              neighborPts.insert(neighborPts.end(), neighborPts_.begin(), neighborPts_.end());
          }
          // if P' is not yet a member of any cluster, add P' to cluster c
          if (!clustered[neighborPts[j]]) {
            clustered[neighborPts[j]] = true;
            clusters.at(c).clusterPoints.push_back(octreeGen->getCloud()->points.at(neighborPts[j]));
          }
        }
        c++;
      }
    }
  }
}

/// Generates a set of clusters.
///@param[in] octreeGen         The octree to be searched.
///@param[in] eps_              The search radius for the octree.
///@param[in] clusters          Vector that stores all the generated clusters.
void dbscan::DBSCAN_Octree_fast2(htr::OctreeGenerator *octreeGen, int minPts) {
  // Clear aux clusters
  clustersAux.clear();

  if (!octreeGen->getCloud()->points.empty()) {
    // Create array of tree iterators
    std::vector<htr::OctreeGenerator::LeafNodeIterator> leafIterators;

    std::vector<int> tempVec;
    cluster tempPointCluster;
    std::vector<std::vector<int>> neighborPts;

    htr::OctreeGenerator::LeafNodeIterator it(octreeGen->getOctree().get());

    while (*(it)) {
      neighborPts.push_back(tempVec);
      clustersAux.push_back(tempPointCluster);

      it.getLeafContainer().getPointIndices(neighborPts.back());
      for (int k = 0; k < neighborPts.back().size(); ++k) {
        clustersAux.back().clusterPoints.push_back(octreeGen->getCloud()->points.at(neighborPts.back().at(k)));
      }
      clustersAux.back().calculateCentroid();

      ++it;
    }
  }
}
}



queue<KeypointCluster> dbscan_classification(int octreeResolution, float eps, int minPtsAux, int minPts, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int show){
//void init(int filter_flag, int octreeResolution, float eps, int minPtsAux, int minPts, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int show){

    queue<KeypointCluster> Keypoint_Cluster_Queue;
    std::vector<htr::Point3D> groupA;
    dbScanSpace::dbscan dbscan;

    /*************************************************************************************************/
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
    /*
    pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter;
    // adding the polynomial func1 to the plotter with [-10, 10] as the range in X axis and "y = x^2" as title
    plotter->addPlotData(doubleVec_normalized, doubleVec_X, "k square distance", vtkChart::LINE, std::vector<char>());
    plotter->spinOnce(300);
    */

    /************************************************************************************************/
    std::cerr << "octreeResolution: " <<octreeResolution<< std::endl;
    std::cerr << "eps: " <<eps<< std::endl;
    std::cerr << "minPtsAux_: " <<minPtsAux<< std::endl;
    std::cerr << "minPts: " <<minPts<< std::endl;
    //----------------------------------------------------------------
    dbscan.init(groupA, cloud, octreeResolution, eps, minPtsAux, minPts); /*RUN DBSCAN*/
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

