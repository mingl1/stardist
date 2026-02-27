#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>

#include "numpy/arrayobject.h"
#include "clipper.hpp"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.141592653589793
#endif


#ifdef _OPENMP
#include <omp.h>
#endif

#include <nanoflann.hpp>

template <typename T>
struct PointCloud2D
{
	struct Point
	{
		T  x,y;
	};
	std::vector<Point>  pts;
	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }
	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline T kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].x;
		else return pts[idx].y;
	}
	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};


inline int round_to_int(float r) {
  return (int)lrint(r);
}



static PyObject* c_star_dist (PyObject *self, PyObject *args) {

  PyArrayObject *src = NULL;
  PyArrayObject *dst = NULL;
  int n_rays;
  int grid_x, grid_y;

  if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &src, &n_rays, &grid_y, &grid_x))
    return NULL;

  npy_intp *dims = PyArray_DIMS(src);

  npy_intp dims_dst[3];
  dims_dst[0] = (dims[0]-1)/grid_y+1;
  dims_dst[1] = (dims[1]-1)/grid_x+1;
  dims_dst[2] = n_rays;

  dst = (PyArrayObject*)PyArray_SimpleNew(3,dims_dst,NPY_FLOAT32);
    
  // # pragma omp parallel for schedule(dynamic)
  // strangely, using schedule(dynamic) leads to segfault on OSX when importing skimage first
#ifdef __APPLE__    
#pragma omp parallel for 
#else
#pragma omp parallel for schedule(dynamic) 
#endif
  for (int i=0; i<dims_dst[0]; i++) {
    for (int j=0; j<dims_dst[1]; j++) {
      const unsigned short value = *(unsigned short *)PyArray_GETPTR2(src,i*grid_y,j*grid_x);
      // background pixel
      if (value == 0) {
        for (int k = 0; k < n_rays; k++) {
          *(float *)PyArray_GETPTR3(dst,i,j,k) = 0;
        }
        // foreground pixel
      } else {
        const float st_rays = (2*M_PI) / n_rays; // step size for ray angles
        for (int k = 0; k < n_rays; k++) {
          const float phi = k*st_rays;
          const float dy = cos(phi);
          const float dx = sin(phi);
          float x = 0, y = 0;
          // move along ray
          while (1) {
            x += dx;
            y += dy;
            const int ii = round_to_int(i*grid_y+x), jj = round_to_int(j*grid_x+y);
            // stop if out of bounds or reaching a pixel with a different value/id
            if (ii < 0 || ii >= dims[0] ||
                jj < 0 || jj >= dims[1] ||
                value != *(unsigned short *)PyArray_GETPTR2(src,ii,jj))
              {
                // small correction as we overshoot the boundary
                const float t_corr = .5f/fmax(fabs(dx),fabs(dy));
                x += (t_corr-1.f)*dx;
                y += (t_corr-1.f)*dy;
                const float dist = sqrt(x*x + y*y);
                *(float *)PyArray_GETPTR3(dst,i,j,k) = dist;
                break;
              }
          }

        }
      }

    }
  }
          
  return PyArray_Return(dst);
}



inline float area_from_path(ClipperLib::Path p) {

  float area = 0;
  const int n = p.size();
  for (int i = 0; i<n; i++) {
    area += p[i].X * p[(i+1)%n].Y -
      p[i].Y * p[(i+1)%n].X;
  }
  area = 0.5 * abs(area);
  return area;
}



inline bool bbox_intersect(const int bbox_a_x1, const int bbox_a_x2,
                           const int bbox_a_y1, const int bbox_a_y2,
                           const int bbox_b_x1, const int bbox_b_x2,
                           const int bbox_b_y1, const int bbox_b_y2) {
  // return !( bbox_b_x1 >  bbox_a_x2 || bbox_a_x1 >  bbox_b_x2 || bbox_b_y1 >  bbox_a_y2 || bbox_a_y1 >  bbox_b_y2 );
  return  ( bbox_b_x1 <= bbox_a_x2 && bbox_a_x1 <= bbox_b_x2 && bbox_b_y1 <= bbox_a_y2 && bbox_a_y1 <= bbox_b_y2 );
}



inline float poly_intersection_area(const ClipperLib::Path poly_a_path, const ClipperLib::Path poly_b_path) {
  ClipperLib::Clipper c;
  ClipperLib::Paths res;
  c.Clear();

  c.AddPath(poly_a_path,ClipperLib::ptClip, true);
  c.AddPath(poly_b_path,ClipperLib::ptSubject, true);
  c.Execute(ClipperLib::ctIntersection, res, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

  float area_inter = 0;
  for (unsigned int r=0; r<res.size(); r++)
    area_inter += area_from_path(res[r]);
  return area_inter;
}

inline float cross2d(const float ay, const float ax, const float by, const float bx) {
  return ay*bx - ax*by;
}

inline float ray_segment_distance(const float origin_y, const float origin_x,
                                  const float dir_y, const float dir_x,
                                  const float ay, const float ax,
                                  const float by, const float bx) {
  const float sy = by-ay;
  const float sx = bx-ax;
  const float denom = cross2d(dir_y, dir_x, sy, sx);
  if (fabs(denom) <= 1.e-8f)
    return -1.f;
  const float wy = ay-origin_y;
  const float wx = ax-origin_x;
  const float t = cross2d(wy, wx, sy, sx) / denom;
  if (t < 0.f)
    return -1.f;
  const float u = cross2d(wy, wx, dir_y, dir_x) / denom;
  if (u < 0.f || u > 1.f)
    return -1.f;
  return t;
}

inline bool analytic_overlap_exceeds(const float *dist_i, const float *dist_j,
                                     const float center_i_y, const float center_i_x,
                                     const float center_j_y, const float center_j_x,
                                     const float *ray_sin, const float *ray_cos,
                                     const int n_rays, const float angle_step,
                                     const float threshold_area) {
  float area_inter = 0.f;
  for (int k = 0; k < n_rays; ++k) {
    const float dir_y = ray_sin[k];
    const float dir_x = ray_cos[k];
    float best_t = -1.f;
    int e_prev = n_rays-1;
    float ay = center_j_y + dist_j[e_prev] * ray_sin[e_prev];
    float ax = center_j_x + dist_j[e_prev] * ray_cos[e_prev];
    for (int e = 0; e < n_rays; ++e) {
      const float by = center_j_y + dist_j[e] * ray_sin[e];
      const float bx = center_j_x + dist_j[e] * ray_cos[e];
      const float t = ray_segment_distance(center_i_y, center_i_x, dir_y, dir_x,
                                           ay, ax, by, bx);
      if (t >= 0.f && (best_t < 0.f || t < best_t))
        best_t = t;
      ay = by;
      ax = bx;
    }
    if (best_t <= 0.f)
      continue;
    const float r = fmin(dist_i[k], best_t);
    if (r <= 0.f)
      continue;
    area_inter += 0.5f * r * r * angle_step;
    if (area_inter > threshold_area)
      return true;
  }
  return false;
}



// polys.shape = (n_polys, 2, n_rays)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept

static PyObject* c_non_max_suppression_inds_old(PyObject *self, PyObject *args) {

  PyArrayObject *polys=NULL, *mapping=NULL, *result=NULL;
  float threshold;
  int max_bbox_search, grid_x, grid_y;
  int verbose;

  if (!PyArg_ParseTuple(args, "O!O!fiiii", &PyArray_Type, &polys, &PyArray_Type, &mapping, &threshold, &max_bbox_search, &grid_y, &grid_x, &verbose))
    return NULL;

  npy_intp *img_dims = PyArray_DIMS(mapping);
  const int height = img_dims[0], width = img_dims[1];

  npy_intp *dims = PyArray_DIMS(polys);
  const int n_polys = dims[0];
  const int n_rays = dims[2];

  int * bbox_x1 = new int[n_polys];
  int * bbox_x2 = new int[n_polys];
  int * bbox_y1 = new int[n_polys];
  int * bbox_y2 = new int[n_polys];

  int max_bbox_size_x = 0;
  int max_bbox_size_y = 0;

  float * areas = new float[n_polys];
  bool * suppressed = new bool[n_polys];
  ClipperLib::Path * poly_paths = new ClipperLib::Path[n_polys];

  int count_suppressed = 0;

  //initialize indices
#pragma omp parallel for
  for (int i=0; i<n_polys; i++) {
    suppressed[i] = false;
  }


  if (verbose){
    printf("Non Maximum Suppression (2D) ++++ \n");
    printf("NMS: n_polys  = %d \nNMS: n_rays   = %d  \nNMS: thresh   = %.3f \nNMS: max_bbox_search = %d \n", n_polys, n_rays, threshold, max_bbox_search);
#ifdef _OPENMP
    printf("NMS: using OpenMP with %d thread(s)\n", omp_get_max_threads());
#endif
  }

  // build polys and areas

  // disable OpenMP  for now, as there is still a race condition (segfaults on OSX)
  for (int i=0; i<n_polys; i++) {
    ClipperLib::Path clip;
    // build clip poly and bounding boxes
    for (int k =0; k<n_rays; k++) {
      int y = *(int *)PyArray_GETPTR3(polys,i,0,k);
      int x = *(int *)PyArray_GETPTR3(polys,i,1,k);
      // printf("%d, %d,  ",y,x);
      if (k==0) {
        bbox_x1[i] = x;
        bbox_x2[i] = x;
        bbox_y1[i] = y;
        bbox_y2[i] = y;
      } else {
        bbox_x1[i] = (x<bbox_x1[i])?x:bbox_x1[i];
        bbox_x2[i] = (x>bbox_x2[i])?x:bbox_x2[i];
        bbox_y1[i] = (y<bbox_y1[i])?y:bbox_y1[i];
        bbox_y2[i] = (y>bbox_y2[i])?y:bbox_y2[i];
      }
      clip<<ClipperLib::IntPoint(x,y);
    }
    if (max_bbox_search) {
      const int bbox_size_x = bbox_x2[i] - bbox_x1[i];
      const int bbox_size_y = bbox_y2[i] - bbox_y1[i];
      if (bbox_size_x > max_bbox_size_x) {
#pragma omp critical (max_x)
        max_bbox_size_x = bbox_size_x;
      }
      if (bbox_size_y > max_bbox_size_y) {
#pragma omp critical (max_y)
        max_bbox_size_y = bbox_size_y;
      }
    }
    poly_paths[i] = clip;
    areas[i] = area_from_path(clip);
  }

  // printf("max_bbox_size_x = %d, max_bbox_size_y = %d\n", max_bbox_size_x, max_bbox_size_y);
  if (verbose)
    printf("NMS: starting suppression loop\n");

  ProgressBar prog("suppressed");

  if (max_bbox_search) {

    // suppress (double loop)
    for (int i=0; i<n_polys-1; i++) {
      if (suppressed[i]) continue;

      if (verbose)
        prog.update(100.*count_suppressed/n_polys);

      // check signals e.g. such that the loop is interruptible
      if (PyErr_CheckSignals()==-1){
        delete [] areas;
        delete [] suppressed;
        delete [] poly_paths;
        delete [] bbox_x1;
        delete [] bbox_x2;
        delete [] bbox_y1;
        delete [] bbox_y2;
        PyErr_SetString(PyExc_KeyboardInterrupt, "interrupted");
        return Py_None;
      }

      const int xs = std::max((bbox_x1[i]-max_bbox_size_x)/grid_x, 0);
      const int xe = std::min((bbox_x2[i]+max_bbox_size_x)/grid_x, width);
      const int ys = std::max((bbox_y1[i]-max_bbox_size_y)/grid_y, 0);
      const int ye = std::min((bbox_y2[i]+max_bbox_size_y)/grid_y, height);

      // printf("%5d [%03d:%03d,%03d:%03d]",i,bbox_x1[i],bbox_x2[i],bbox_y1[i],bbox_y2[i]);
      // printf(" - search area [%03d:%03d,%03d:%03d]\n",xs,xe,ys,ye);

      // cf. https://github.com/peterwittek/somoclu/issues/111
#ifdef _WIN32
#pragma omp parallel for schedule(dynamic) reduction(+:count_suppressed)
#else
#ifdef __APPLE__
#pragma omp parallel for collapse(2) reduction(+:count_suppressed)
#else
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:count_suppressed)
#endif
#endif
      for (int jj=ys; jj<ye; jj++) for (int ii=xs; ii<xe; ii++) {
          // j is the id of the score-sorted polygon at coordinate (ii,jj)
          const int j = *(int *)PyArray_GETPTR2(mapping,jj,ii);
          // if (j<0) continue;  // polygon not even a candidate (check redundant because of next line)
          if (j<=i) continue; // polygon has higher score (i.e. lower id) than "suppressor polygon" i
          if (suppressed[j]) continue;
          // skip if bounding boxes are not even intersecting
          if (!bbox_intersect(bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j], bbox_y1[j], bbox_y2[j]))
            continue;

          const float area_inter = poly_intersection_area(poly_paths[i], poly_paths[j]);
          const float overlap = area_inter / fmin( areas[i]+1.e-10, areas[j]+1.e-10 );
          if (overlap > threshold){
            count_suppressed +=1;
            suppressed[j] = true;
          }
        }
    }

  } else {

    // suppress (double loop)
    for (int i=0; i<n_polys-1; i++) {
      if (suppressed[i]) continue;

      if (verbose)
        prog.update(100.*count_suppressed/n_polys);

      // check signals e.g. such that the loop is interruptible
      if (PyErr_CheckSignals()==-1){
        delete [] areas;
        delete [] suppressed;
        delete [] poly_paths;
        delete [] bbox_x1;
        delete [] bbox_x2;
        delete [] bbox_y1;
        delete [] bbox_y2;
        PyErr_SetString(PyExc_KeyboardInterrupt, "interrupted");
        return Py_None;
      }

      // printf("%5d [%03d:%03d,%03d:%03d]\n",i,bbox_x1[i],bbox_x2[i],bbox_y1[i],bbox_y2[i]);

#ifdef __APPLE__
#pragma omp parallel for reduction(+:count_suppressed)
#else
#pragma omp parallel for schedule(dynamic) reduction(+:count_suppressed)
#endif
      for (int j=i+1; j<n_polys; j++) {
        if (suppressed[j]) continue;
        // skip if bounding boxes are not even intersecting
        if (!bbox_intersect(bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j], bbox_y1[j], bbox_y2[j]))
          continue;

        const float area_inter = poly_intersection_area(poly_paths[i], poly_paths[j]);
        const float overlap = area_inter / fmin( areas[i]+1.e-10, areas[j]+1.e-10 );
        if (overlap > threshold){
          count_suppressed +=1;
          suppressed[j] = true;
        }

      }
    }

  }

  if (verbose)
    prog.finish();

  if (verbose){
    printf("NMS: Suppressed polygons:   %8d / %d  (%.2f %%)\n", count_suppressed,n_polys,100*(float)count_suppressed/n_polys);
  }

  npy_intp dims_result[1];
  dims_result[0] = n_polys;

  result = (PyArrayObject*)PyArray_SimpleNew(1,dims_result,NPY_BOOL);

  for (int i=0; i<n_polys;i++)
    *(bool *)PyArray_GETPTR1(result,i) = !suppressed[i];

  delete [] areas;
  delete [] suppressed;
  delete [] poly_paths;
  delete [] bbox_x1;
  delete [] bbox_x2;
  delete [] bbox_y1;
  delete [] bbox_y2;

  return PyArray_Return(result);
}


  
static PyObject* c_non_max_suppression_inds(PyObject *self, PyObject *args) {

  PyArrayObject *dist=NULL, *points_arr=NULL, *result=NULL;
  float threshold;
  int verbose, use_kdtree, use_bbox, use_analytic;
  use_analytic = 0;

  if (!PyArg_ParseTuple(args, "O!O!iiif|i", &PyArray_Type, &dist, &PyArray_Type, &points_arr ,
                        &use_kdtree, &use_bbox, &verbose, &threshold, &use_analytic))
    return NULL;

  const float * const points = (float*) PyArray_DATA(points_arr);

  npy_intp *dims = PyArray_DIMS(dist);
  const int n_polys = dims[0];
  const int n_rays = dims[1];

  float * bbox_x1 = new float[n_polys];
  float * bbox_x2 = new float[n_polys];
  float * bbox_y1 = new float[n_polys];
  float * bbox_y2 = new float[n_polys];
  float * radius_outer = new float[n_polys];
  float * radius_inner = new float[n_polys];

  float * areas = new float[n_polys];
  bool * suppressed = new bool[n_polys];
  ClipperLib::Path * poly_paths = new ClipperLib::Path[n_polys];

  const float ANGLE_PI = 2*M_PI/n_rays;
  float * ray_sin = new float[n_rays];
  float * ray_cos = new float[n_rays];
  for (int k = 0; k < n_rays; ++k){
    ray_sin[k] = sin(ANGLE_PI*k);
    ray_cos[k] = cos(ANGLE_PI*k);
  }
  
  int count_suppressed = 0;

  //initialize indices
#pragma omp parallel for
  for (int i=0; i<n_polys; i++) {
    suppressed[i] = false;
  }


  if (verbose){
    printf("Non Maximum Suppression (2D) ++++ \n");
    printf("NMS: n_polys    = %d \nNMS: n_rays     = %d  \nNMS: thresh     = %.3f \nNMS: use_bbox   = %d\nNMS: use_kdtree = %d\nNMS: use_analytic = %d\n", n_polys, n_rays, threshold, use_bbox, use_kdtree, use_analytic);
#ifdef _OPENMP
    printf("NMS: using OpenMP with %d thread(s)\n", omp_get_max_threads());
#endif
  }

  // build polys and areas

  // disable OpenMP  for now, as there is still a race condition (segfaults on OSX)
  // #pragma omp parallel for
  for (int i=0; i<n_polys; i++) {
    ClipperLib::Path clip;
    // build clip poly and bounding boxes
    // const int py = *(int *)PyArray_GETPTR2(points,i,0);
    // const int px = *(int *)PyArray_GETPTR2(points,i,1);
    const float py = points[2*i];
    const float px = points[2*i+1];
    float max_radius_outer = 0;
    float min_radius_inner = 0;
    for (int k =0; k<n_rays; k++) {
      // int y = *(int *)PyArray_GETPTR3(polys,i,0,k);
      // int x = *(int *)PyArray_GETPTR3(polys,i,1,k);
      const float d = *(float*)PyArray_GETPTR2(dist,i,k);  
      const float y = (float)(py+d*ray_sin[k]);
      const float x = (float)(px+d*ray_cos[k]);


      // printf("%d, %d,  ",y, x);
      
      if (k==0) {
        bbox_x1[i] = x;
        bbox_x2[i] = x;
        bbox_y1[i] = y;
        bbox_y2[i] = y;
      } else {
        bbox_x1[i] = (x<bbox_x1[i])?x:bbox_x1[i];
        bbox_x2[i] = (x>bbox_x2[i])?x:bbox_x2[i];
        bbox_y1[i] = (y<bbox_y1[i])?y:bbox_y1[i];
        bbox_y2[i] = (y>bbox_y2[i])?y:bbox_y2[i];
      }
      clip<<ClipperLib::IntPoint(x,y);

      max_radius_outer = fmax(d,max_radius_outer);
      min_radius_inner = (k==0) ? d : fmin(d,min_radius_inner);
    }
    radius_outer[i] = max_radius_outer;
    radius_inner[i] = min_radius_inner;
    
    poly_paths[i] = clip;
    areas[i] = area_from_path(clip);
  }



  // build kdtree

  PointCloud2D<float> cloud;
  nanoflann::SearchParams params;
  std::vector<std::pair<size_t,float>> results;
  float * radius_outer_suffix_max = new float[n_polys+1];

  cloud.pts.resize(n_polys);
  for (long i = 0; i < n_polys; i++){
    cloud.pts[i].x = points[2*i];
    cloud.pts[i].y = points[2*i+1];
  }
  radius_outer_suffix_max[n_polys] = 0;
  for (int i = n_polys-1; i >= 0; --i)
    radius_outer_suffix_max[i] = fmax(radius_outer[i], radius_outer_suffix_max[i+1]);
  
  // construct a kd-tree:
  typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud2D<float>> ,
    PointCloud2D<float>,2> my_kd_tree_t;

  //build the index from points
  my_kd_tree_t  index(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

  if (use_kdtree){
    if (verbose){
      printf("NMS: building kdtree...\n");
      fflush(stdout);
    }
    index.buildIndex();
  }


  
  // printf("max_bbox_size_x = %d, max_bbox_size_y = %d\n", max_bbox_size_x, max_bbox_size_y);
  if (verbose)
    printf("NMS: starting suppression loop\n");

  ProgressBar prog("suppressed");

  
  // suppress (double loop)
  for (int i=0; i<n_polys-1; i++) {
    if (suppressed[i]) continue;

    if (verbose)
      prog.update(100.*count_suppressed/n_polys);

    // check signals e.g. such that the loop is interruptible
    if (PyErr_CheckSignals()==-1){
      delete [] areas;
      delete [] suppressed;
      delete [] poly_paths;
      delete [] bbox_x1;
      delete [] bbox_x2;
      delete [] bbox_y1;
      delete [] bbox_y2;
      delete [] radius_outer;
      delete [] radius_inner;
      delete [] radius_outer_suffix_max;
      delete [] ray_sin;
      delete [] ray_cos;
      PyErr_SetString(PyExc_KeyboardInterrupt, "interrupted");
      return Py_None;
    }
    // printf("%.2f %.2f\n",points[2*i],points[2*i+1]);

    if (use_kdtree){
      results.clear();
      const float search_radius = radius_outer[i] + radius_outer_suffix_max[i+1];
      if (search_radius > 0)
        index.radiusSearch(&points[2*i],
                           search_radius*search_radius,
                           results, params);
    } else {
      results.clear();
      results.resize(n_polys-i);
      for (size_t n = 0; n < results.size(); ++n)
        results[n].first = i+n;
    }
    

    const float * const dist_i_ptr = (float*)PyArray_GETPTR2(dist,i,0);

    // inner loop 
    // remove  schedule(dynamic) on OSX as it leads to segfaults sometimes (TODO)
#ifdef __APPLE__    
#pragma omp parallel for reduction(+:count_suppressed)   shared(suppressed) 
#else
#pragma omp parallel for schedule(dynamic) reduction(+:count_suppressed)   shared(suppressed)
#endif

    for (size_t neigh=0; neigh<results.size(); neigh++) {
    // for (int j=i+1; j<n_polys; j++) {

      long j = results[neigh].first;
      // printf("%d %d",i,j);
      
      if ((suppressed[j]) || (j<=i))
        continue;

      const float cy = points[2*i] - points[2*j];
      const float cx = points[2*i+1] - points[2*j+1];
      const float dist2 = cy*cy + cx*cx;

      const float rsum = radius_outer[i] + radius_outer[j];
      if (dist2 > rsum*rsum)
        continue;

      const float inner2 = radius_inner[i] * radius_inner[i];
      if (dist2 < inner2){
        count_suppressed +=1;
#pragma omp atomic write
        suppressed[j] = true;
        continue;
      }
      
      // skip if bounding boxes are not even intersecting
      if ((use_bbox) && (!bbox_intersect(bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j], bbox_y1[j], bbox_y2[j])))
        continue;

      bool suppress = false;
      const float min_area = fmin(areas[i]+1.e-10f, areas[j]+1.e-10f);
      if (use_analytic) {
        const float * const dist_j_ptr = (float*)PyArray_GETPTR2(dist,j,0);
        suppress = analytic_overlap_exceeds(dist_i_ptr, dist_j_ptr,
                                            points[2*i], points[2*i+1],
                                            points[2*j], points[2*j+1],
                                            ray_sin, ray_cos,
                                            n_rays, ANGLE_PI,
                                            threshold * min_area);
      } else {
        const float area_inter = poly_intersection_area(poly_paths[i], poly_paths[j]);
        suppress = (area_inter / min_area) > threshold;
      }
      if (suppress){
        count_suppressed +=1;
#pragma omp atomic write
        suppressed[j] = true;
          
      }

    }
  }

  if (verbose)
    prog.finish();

  if (verbose){
    printf("NMS: Suppressed polygons:   %8d / %d  (%.2f %%)\n", count_suppressed,n_polys,100*(float)count_suppressed/n_polys);
  }

  npy_intp dims_result[1];
  dims_result[0] = n_polys;

  result = (PyArrayObject*)PyArray_SimpleNew(1,dims_result,NPY_BOOL);

  for (int i=0; i<n_polys;i++)
    *(bool *)PyArray_GETPTR1(result,i) = !suppressed[i];

  delete [] areas;
  delete [] suppressed;
  delete [] poly_paths;
  delete [] bbox_x1;
  delete [] bbox_x2;
  delete [] bbox_y1;
  delete [] bbox_y2;
  delete [] radius_outer;
  delete [] radius_inner;
  delete [] radius_outer_suffix_max;
  delete [] ray_sin;
  delete [] ray_cos;

  return PyArray_Return(result);
}


//------------------------------------------------------------------------


static struct PyMethodDef methods[] = {
                                       {"c_non_max_suppression_inds_old",
                                        c_non_max_suppression_inds_old,
                                        METH_VARARGS, "non-maximum suppression"},
                                       {"c_non_max_suppression_inds",
                                        c_non_max_suppression_inds,
                                        METH_VARARGS, "non-maximum suppression"},
                                       {"c_star_dist",
                                        c_star_dist,
                                        METH_VARARGS, "star dist calculation"},
                                       {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "stardist2d",
                                       NULL,
                                       -1,
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_stardist2d(void) {
  import_array();
  return PyModule_Create(&moduledef);
}
