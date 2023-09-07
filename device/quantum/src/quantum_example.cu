/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "quantum_example.cuh"
#include <complex.h>
#include "Python.h"
#include "numpy/arrayobject.h"

#if defined(TARGET_DEVICE_CUDA)
#include <cuComplex.h>
#include <custatevec.h>
#endif

INSTANTIATE_ALGORITHM(quantum::quantum_t)

int init_np()
{
  import_array();
  return 0;
}

void quantum::quantum_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{

  /*
   * Initialize python interpreter and load module
   */
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }

  /*
   * Load the module and check for errors
   */
  PyObject* module_name = PyUnicode_FromString("quantum_circuit");
  PyObject* module = PyImport_Import(module_name);
  // TODO: Fix any errors imporitng not outputting erros here.
  if (!module) {
    std::cout << "quantum_circuit.py couldn't be imported. Ensure this file is in a directory findable by python. "
                 "(e.g., in your PYTHONPATH)"
              << std::endl;
    PyErr_Print();
    return;
  }
  PyObject* module_dict = PyModule_GetDict(module);

  /*
   * Load Davides expected input into vector
   */
  PyObject* result = PyList_New(0);

  const auto a = Allen::ArgumentOperations::make_host_buffer<dev_velo_cluster_container_t>(arguments, context);
  const auto offsets_estimated_input_size = Allen::ArgumentOperations::make_host_buffer<dev_offsets_estimated_input_size_t>(arguments, context);
  const auto module_cluster_num = Allen::ArgumentOperations::make_host_buffer<dev_module_cluster_num_t>(arguments, context);

  const auto velo_cluster_container =
    Velo::ConstClusters {a.data(), Allen::ArgumentOperations::first<host_total_number_of_velo_clusters_t>(arguments)};
  std::cout << "Building hit list" << std::endl;
  for (unsigned event_number = 0; event_number < Allen::ArgumentOperations::first<host_number_of_events_t>(arguments);
       ++event_number) {
    const auto event_number_of_hits =
      offsets_estimated_input_size[(event_number + 1) * Velo::Constants::n_module_pairs] -
      offsets_estimated_input_size[event_number * Velo::Constants::n_module_pairs];
    if (event_number_of_hits > 0) {
      for (unsigned i = 0; i < Velo::Constants::n_module_pairs; ++i) {
        const auto module_hit_start = offsets_estimated_input_size[event_number * Velo::Constants::n_module_pairs + i];
        const auto module_hit_num = module_cluster_num[event_number * Velo::Constants::n_module_pairs + i];

        for (unsigned hit_number = 0; hit_number < module_hit_num; ++hit_number) {
          const auto hit_index = module_hit_start + hit_number;

          // Prepare main vector with data
          PyObject* hit = PyList_New(0);
          PyList_Append(hit, PyFloat_FromDouble(velo_cluster_container.x(hit_index)));
          PyList_Append(hit, PyFloat_FromDouble(velo_cluster_container.y(hit_index)));
          PyList_Append(hit, PyFloat_FromDouble(velo_cluster_container.z(hit_index)));
          PyList_Append(hit, PyLong_FromLong(velo_cluster_container.id(hit_index)));
          PyList_Append(hit, PyLong_FromLong(i));
          PyList_Append(result, hit);
        }
      }
    }
  }

  /*
   * Set MCTS data
   */
  std::cout << "MCTS  list" << std::endl;
  PyObject* mcts_data = PyList_New(0);
  const auto mc_data = *first<host_mc_events_t>(arguments);
  for (const auto& event : mc_data) {
    for (const auto& mcp : event.m_mcps) {
      PyObject* track_data = PyList_New(0);
      for (const auto& lhcb_id : mcp.hits) {
        PyList_Append(track_data, PyLong_FromLong(lhcb_id));
      }
      PyList_Append(mcts_data, track_data);
    }
  }

  /*
   * Prepare function arguments
   */
  std::cout << "Building function argument list" << std::endl;
  PyObject* func_args = PyTuple_New(2);
  PyTuple_SetItem(func_args, 0, result);
  PyTuple_SetItem(func_args, 1, mcts_data);

  PyObject* func = PyDict_GetItemString(module_dict, (char*) "circuit");
  std::cout << PyCallable_Check(func) << std::endl;
  if (PyCallable_Check(func)) {
    printf("Calling python function\n");
    PyObject* ret = PyObject_CallObject(func, func_args);
    //    PyArrayObject* np_ret = reinterpret_cast<PyArrayObject*>(ret);
    //    std::cout << np_ret << std::endl;
    //    npy_intp width = PyArray_DIM(np_ret, 0);
    //    npy_intp height = PyArray_DIM(np_ret, 1);
    //    std::cout << "GOt matrix of size" << width << "x" << height << std::endl;
    //    std::complex<double>* c_out = reinterpret_cast<std::complex<double>*>(PyArray_DATA(np_ret));
    //    std::cout << c_out[0] << std::endl;
  }
#if defined(TARGET_DEVICE_CUDA)
  const int nIndexBits = 3;
  const int nSvSize = (1 << nIndexBits);
  const int nTargets = 1;
  const int nControls = 2;
  const int adjoint = 0;

  int targets[] = {2};
  int controls[] = {0, 1};

  cuDoubleComplex h_sv[] = {
    {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
  cuDoubleComplex h_sv_result[] = {
    {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.4, 0.5}, {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.1, 0.2}};

  cuDoubleComplex matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  cuDoubleComplex* d_sv;
  cudaMalloc((void**) &d_sv, nSvSize * sizeof(cuDoubleComplex));

  cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  //--------------------------------------------------------------------------

  // custatevec handle initialization
  custatevecHandle_t handle;

  custatevecCreate(&handle);

  void* extraWorkspace = nullptr;
  size_t extraWorkspaceSizeInBytes = 0;

  // check the size of external workspace
  custatevecApplyMatrixGetWorkspaceSize(
    handle,
    CUDA_C_64F,
    nIndexBits,
    matrix,
    CUDA_C_64F,
    CUSTATEVEC_MATRIX_LAYOUT_ROW,
    adjoint,
    nTargets,
    nControls,
    CUSTATEVEC_COMPUTE_64F,
    &extraWorkspaceSizeInBytes);

  // allocate external workspace if necessary
  if (extraWorkspaceSizeInBytes > 0) cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes);

  // apply gate
  custatevecApplyMatrix(
    handle,
    d_sv,
    CUDA_C_64F,
    nIndexBits,
    matrix,
    CUDA_C_64F,
    CUSTATEVEC_MATRIX_LAYOUT_ROW,
    adjoint,
    targets,
    nTargets,
    controls,
    nullptr,
    nControls,
    CUSTATEVEC_COMPUTE_64F,
    extraWorkspace,
    extraWorkspaceSizeInBytes);

  // destroy handle
  custatevecDestroy(handle);

  //--------------------------------------------------------------------------

  cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i < nSvSize; i++) {
    if ((h_sv[i].x != h_sv_result[i].x) || (h_sv[i].y != h_sv_result[i].y)) {
      correct = false;
      break;
    }
  }

  if (correct)
    printf("example PASSED\n");
  else
    printf("example FAILED: wrong result\n");

  cudaFree(d_sv);
  if (extraWorkspaceSizeInBytes) cudaFree(extraWorkspace);
#endif
}

///**
// * @brief SAXPY example algorithm
// * @detail Calculates for every event y = a*x + x, where x is the number of velo tracks in one event
// */
//__device__ void quantum::quantum(quantum::Parameters parameters)
//{
//  const auto number_of_events = parameters.dev_number_of_events[0];
//  for (unsigned event_number = threadIdx.x; event_number < number_of_events; event_number += blockDim.x) {
//    Velo::Consolidated::ConstTracks velo_tracks {
//      parameters.dev_atomics_velo, parameters.dev_velo_track_hit_number, event_number, number_of_events};
//    const unsigned number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
//
//    parameters.dev_saxpy_output[event_number] =
//      parameters.saxpy_scale_factor * number_of_tracks_event + number_of_tracks_event;
//  }
//}
