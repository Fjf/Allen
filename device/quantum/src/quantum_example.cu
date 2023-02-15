/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "quantum_example.cuh"

#include "Python.h"

#if defined(TARGET_DEVICE_CUDA)
#include <cuComplex.h>
#include <custatevec.h>
#endif

INSTANTIATE_ALGORITHM(quantum::quantum_t)

void quantum::quantum_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_saxpy_output_t>(arguments, first<host_number_of_events_t>(arguments));
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
  Py_Initialize();

  PyObject* module_name = PyUnicode_FromString("quantum_circuit");
  PyObject* module = PyImport_Import(module_name);
  if (!module) {
    std::cout << "quantum_circuit.py couldn't be imported. Ensure this file is in a directory findable by python. "
                 "(e.g., in your PYTHONPATH)"
              << std::endl;
    return;
  }

  PyObject* module_dict = PyModule_GetDict(module);

  /*
   * Load Davides expected input into vector
   */
  PyObject* result = PyList_New(0);
  for (int i = 0; i < 100; i++) {
    PyList_Append(result, PyLong_FromLong(i));
  }

  PyObject* func_args = PyTuple_New(1);
  PyTuple_SetItem(func_args, 0, result);
  PyObject* func = PyDict_GetItemString(module_dict, (char*) "circuit");
  if (PyCallable_Check(func)) {
    PyObject_CallObject(func, func_args);
  }
  Py_Finalize();

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

/**
 * @brief SAXPY example algorithm
 * @detail Calculates for every event y = a*x + x, where x is the number of velo tracks in one event
 */
__device__ void quantum::quantum(quantum::Parameters parameters)
{
  const auto number_of_events = parameters.dev_number_of_events[0];
  for (unsigned event_number = threadIdx.x; event_number < number_of_events; event_number += blockDim.x) {
    Velo::Consolidated::ConstTracks velo_tracks {
      parameters.dev_atomics_velo, parameters.dev_velo_track_hit_number, event_number, number_of_events};
    const unsigned number_of_tracks_event = velo_tracks.number_of_tracks(event_number);

    parameters.dev_saxpy_output[event_number] =
      parameters.saxpy_scale_factor * number_of_tracks_event + number_of_tracks_event;
  }
}
