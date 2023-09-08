/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "quantum_example.cuh"
#include <complex.h>
#include "Python.h"
#include <bits/stdc++.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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
  // TODO: Fix any errors importing not outputting errors here.
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
  const auto offsets_estimated_input_size =
    Allen::ArgumentOperations::make_host_buffer<dev_offsets_estimated_input_size_t>(arguments, context);
  const auto module_cluster_num =
    Allen::ArgumentOperations::make_host_buffer<dev_module_cluster_num_t>(arguments, context);

  const auto velo_cluster_container =
    Velo::ConstClusters {a.data(), Allen::ArgumentOperations::first<host_total_number_of_velo_clusters_t>(arguments)};
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
  PyObject* func_args = PyTuple_New(2);
  PyTuple_SetItem(func_args, 0, result);
  PyTuple_SetItem(func_args, 1, mcts_data);

  PyObject* func = PyDict_GetItemString(module_dict, (char*) "circuit");

  // Prepare return value storage.
  npy_intp height;
  PyArrayObject* np_ret;
  long len_b;

  if (PyCallable_Check(func)) {
    PyObject* return_tuple = PyObject_CallObject(func, func_args);
    if (return_tuple == NULL) {
      PyErr_Print();
      return;
    }
    len_b = PyLong_AsLong(PyTuple_GetItem(return_tuple, 1));
    np_ret = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(return_tuple, 0));
    height = PyArray_DIM(np_ret, 1);
  } else {
    std::cout << "Python function returned error."
              << std::endl;
    PyErr_Print();
    return;
  }
#if defined(TARGET_DEVICE_CUDA)
  const int nSvSize = height;
  const int nIndexBits = log2(nSvSize);
  const int nTargets = nIndexBits;
  const int nControls = 0;
  const int adjoint = 0;

  // Create array to define all qubits as targets.
  int* targets = (int*)malloc(height * sizeof(cuDoubleComplex));
  for (int i = 0; i < nIndexBits; i++) targets[i] = i;
  int controls[] = {};

  // Initialize zero-initialized array of doubles with the first real element set to 1 as statevector.
  cuDoubleComplex* h_sv = (cuDoubleComplex*)calloc(nSvSize, sizeof(cuDoubleComplex));
  h_sv[0].x = 1;

  // Copy data to device.
  cuDoubleComplex* d_sv;
  cudaMalloc((void**) &d_sv, nSvSize * sizeof(cuDoubleComplex));
  cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  // Convert data to cuDoubleComplex
  cuDoubleComplex* matrix = reinterpret_cast<cuDoubleComplex*>(PyArray_DATA(np_ret));

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

  int post_select_qubit = int(log2(float(nSvSize)) - 1.0);
  int solution_length = len_b;
  int base = 1 << post_select_qubit;
  std::vector<double> solution_vector;

  double euclidian_sum;
  for (int i = nSvSize / 2; i < nSvSize; i++) {
    // Compute euclidian sum
    cuDoubleComplex a =  h_sv[i];
    euclidian_sum += (a.x * a.x) + (a.y * a.y);  // Sum of real^2 + imag^2
  }
  euclidian_sum = std::sqrt(euclidian_sum);

  double sum;
  for (int i = 0; i < solution_length; i++) {
    // Create solution vector subset
    cuDoubleComplex a =  h_sv[i + base];
    solution_vector.push_back(a.x);

    sum += a.x * a.x;  // Sum of real parts of sol_vector
  }
  sum = std::sqrt(sum);


  // The rest
  for (int i = 0; i < solution_vector.size(); i++) {
    float magic_value = 2.; // TODO: This is the lowest eigen value from A, we should calculate this in a smart way
    solution_vector[i] = (solution_vector[i] / sum) * euclidian_sum * std::sqrt(float(len_b)) / magic_value;
    std::cout << solution_vector[i] << ",";
  }

  std::cout << std::endl;


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
