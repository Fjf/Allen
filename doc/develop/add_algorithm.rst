.. _add_Allen_algorithm:

Adding a new device algorithm
====================================

This tutorial will guide you through adding a new device algorithm to the `Allen` framework.

SAXPY
^^^^^^^^^^^^

Writing functions to be executed in the device in `Allen` is literally the same as writing a CUDA kernel. Therefore, you may use any existing tutorial or documentation on how to write good CUDA code.

Writing a device algorithm in the `Allen` framework has been made to resemble the Gaudi syntax, where possible.

Let's assume that we want to run the following classic `SAXPY` CUDA kernel, taken out from |cuda_devblogs|.

.. |cuda_devblogs| raw:: html

   <a href="https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/" target="_blank">this website</a>

.. code-block::c++

  __global__ void saxpy(float *x, float *y, int n, float a) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
  }

Adding the device algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We want to add the algorithm to a specific folder inside the `device` folder::

  device
  ├── associate
  │   ├── CMakeLists.txt
  │   ├── include
  │   │   ├── AssociateConstants.cuh
  │   │   └── VeloPVIP.cuh
  │   └── src
  │       └── VeloPVIP.cu
  ├── CMakeLists.txt
  ├── event_model
  │   ├── associate
  │   │   └── include
  │   │       └── AssociateConsolidated.cuh
  │   ├── common
  │   │   └── include
  │   │       ├── ConsolidatedTypes.cuh
  │   │       └── States.cuh
  ...

Let's create a new folder inside the `device` directory named `example`. We need to modify `device/CMakeLists.txt` to reflect this:

.. code-block:: cmake

  add_subdirectory(raw_banks)
  add_subdirectory(example)

Inside the `example` folder we will create the following structure:

  ├── example
  │   ├── CMakeLists.txt
  │   ├── include
  │   │   └── Saxpy_example.cuh
  │   └── src
  │       └── Saxpy_example.cu

The newly created `example/CMakeLists.txt` file should reflect the project we are creating. We can do that by populating it like so:

.. code-block:: cmake

  file(GLOB saxpy_sources "src/*cu")

  include_directories(include)
  include_directories(${PROJECT_SOURCE_DIR}/device/velo/common/include)
  include_directories(${PROJECT_SOURCE_DIR}/device/event_model/common/include)
  include_directories(${PROJECT_SOURCE_DIR}/device/event_model/velo/include)
  include_directories(${PROJECT_SOURCE_DIR}/main/include)
  include_directories(${PROJECT_SOURCE_DIR}/stream/gear/include)
  include_directories(${PROJECT_SOURCE_DIR}/stream/sequence/include)

  allen_add_device_library(Examples STATIC
    ${saxpy_sources}
  )

The includes of Velo and event model files are only necessary because we will use the number of Velo tracks per event as an input to the saxpy algorithm.
If your new algorithm does not use any Velo related objects, this is not necessary.

The includes of main, gear and sequence are required for any new algorithm in Allen.

Link the new library "Examples" to the stream library in `stream/CMakeLists.txt`:

.. code-block:: cmake

  target_link_libraries(Stream PRIVATE
    CudaCommon
    Associate
    Velo
    AllenPatPV
    PV_beamline
    HostClustering
    HostPrefixSum
    UT
    Kalman
    VertexFitter
    RawBanks
    Selections
    SciFi
    HostGEC
    Muon
    Utils
    Examples
    HostDataProvider
    HostInitEventList)

Next, we create the header file for our algorithm `SAXPY_example.cuh`, which is similar to an algorithm definition in Gaudi: inputs, outputs and properties are defined, as well as the algorithm function itself and an operator calling the function.

There are slight differences to Gaudi, since we want to be able to run the algorithm on a GPU.
The full file is under `here <https://gitlab.cern.ch/lhcb/Allen/-/blob/master/device/example/include/SAXPY_example.cuh>`_. Let's take a look at the components:

.. code-block:: c++

  #pragma once

  #include "VeloConsolidated.cuh"
  #include "AlgorithmTypes.cuh"

The Velo include is only required if Velo objects are used in the algorithm. `DeviceAlgorithm.cuh` defines class `DeviceAlgorithm` and some other handy resources.

Parameters and properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: c++

  namespace saxpy {
    struct Parameters {
      HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
      DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
      DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_atomics_velo;
      DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned) dev_velo_track_hit_number;
      DEVICE_OUTPUT(dev_saxpy_output_t, float) dev_saxpy_output;
      PROPERTY(saxpy_scale_factor_t, "saxpy_scale_factor", "scale factor a used in a*x + y", float) saxpy_scale_factor;
      PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
    };
  }

In the `saxpy` namespace the parameters and properties are specified. Parameters *scope* can either be the host or the device, and they can either be inputs or outputs. Parameters should be defined with the following convention::

    <scope>_<io>(<name>, <type>) <identifier>;

Some parameter examples:

.. code-block:: c++

   DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_atomics_velo;

Defines an input on the *device memory*. It has a name `dev_offsets_all_velo_tracks_t`, which can be later used to identify this argument. It is of type _unsigned_, which means the memory location named `dev_offsets_all_velo_tracks_t` holds `unsigned`s. The *io* and the *type* define the underlying type of the instance to be `<io> <type> *` -- in this case, since it is an input type, `const unsigned*`. Its identifier is `dev_atomics_velo`.

.. code-block:: c++

   DEVICE_OUTPUT(dev_saxpy_output_t, float) dev_saxpy_output;

Defines an output parameter on *device memory*, with name `dev_saxpy_output_t` and identifier `dev_saxpy_output`. Its underlying type is `float*`.

.. code-block:: c++

   HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;

Defines an input parameter on *host memory*, with name `host_number_of_events_t` and identifier `host_number_of_events`. Its underlying type is `const unsigned*`.

.. code-block:: c++

   DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;

Defines an input parameter on *device memory*, with name `dev_number_of_events_t` and identifier `dev_number_of_events`. Its underlying type is `const unsigned*`.

Properties of algorithms define constants and can be configured prior to running the application. They are defined in two parts. First, they should be defined in the `DEFINE_PARAMETERS` macro following the convention::

    PROPERTY(<name>, <key>, <description>, <type>) <identifier>;

For example like this:

.. code-block:: c++

   PROPERTY(saxpy_scale_factor_t, "saxpy_scale_factor", "scale factor a used in a*x + y", float) saxpy_scale_factor

Property with name `saxpy_scale_factor_t` is of type `float`. It will be accessible through key `"saxpy_scale_factor"` in a python configuration file, and it has description `"scale factor a used in a*x + y"`. Its identifier is `saxpy_scale_factor`. Properties *underlying type* is always the same as their type, so in this case `float`.

And second, properties should be defined inside the algorithm struct as follows::

    Property<_name_> _internal_name_ {this, _default_value_};

In the case of saxpy:

.. code-block:: c++

  private:
    Property<saxpy_scale_factor_t> m_saxpy_factor {this, 2.f};
    Property<block_dim_t> m_block_dim {this, {{32, 1, 1}}};

Views
^^^^^^^^^^
A view is a parameter that is linked to other parameters. It extends the lifetime of the parameters it is linked to, ensuring that the data it links to will not be freed.

A view can be defined like a parameter with additional types::

    <scope>_<io>(<name>, <type>, <linked_lifetime_type_1>, <linked_lifetime_type_2>, ...) <identifier>;

Here is a working example:

.. code-block:: c++

    DEVICE_OUTPUT(
      dev_velo_clusters_t,
      Velo::Clusters, dev_velo_cluster_container_t, dev_module_cluster_num_t, dev_number_of_events_t)
    dev_velo_clusters;

The type `dev_velo_clusters_t` is defined to be of type `Velo::Clusters`, with its lifetime linked to types `dev_velo_cluster_container_t, dev_module_cluster_num_t, dev_number_of_events_t`. That is, if `dev_velo_clusters_t` is used in a subsequent algorithm as an input, the parameters `dev_velo_cluster_container_t, dev_module_cluster_num_t, dev_number_of_events_t` are guaranteed to be in memory.

This type can be used just like any other type:

.. code-block:: c++

  auto velo_cluster_container = Velo::Clusters {parameters.dev_velo_cluster_container, estimated_number_of_clusters};
  parameters.dev_velo_clusters[event_number] = velo_cluster_container;

And subsequent algorithms can request it with no need to specify it as a view anymore:

.. code-block:: c++

  DEVICE_INPUT(dev_velo_clusters_t, Velo::Clusters) dev_velo_clusters;

The reason these two types are compatible is because the `Allen underlying type` of both the view and non-view parameter is `Velo::Clusters`.

Defining an algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^

SAXPY_example.cuh
-----------------------
An algorithm is defined by a `struct` (or `class`) that inherits from either `HostAlgorithm` or `DeviceAlgorithm`. In addition, it is convenient to also inherit from `Parameters`, to be able to easily access *identifiers* of parameters and properties. The struct identifier is the name of the algorithm.

An algorithm must define **two methods**: `set_arguments_size` and `operator()`. Their signatures are as follows:

.. code-block:: c++

  struct saxpy_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters>,
      const RuntimeOptions&,
      const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<saxpy_scale_factor_t> m_saxpy_factor {this, 2.f};
    Property<block_dim_t> m_block_dim {this, {{32, 1, 1}}};
  };

An algorithm `saxpy_t` has been declared. It is a `DeviceAlgorithm`, and for convenience it inherits from the previously defined `Parameters`. It defines two methods, `set_arguments_size` and `operator()` with the above predefined signatures. The algorithm declaration ends with the `private:` block for the properties mentioned before.

Since this is a DeviceAlgorithm, one would like the work to actually be done on the device. In order to run code on the device, a *global kernel* has to be defined. The syntax used is standard CUDA:

.. code-block:: c++

  __global__ void saxpy(Parameters);

SAXPY_example.cu
--------------------
The source file of SAXPY should define `set_arguments_size`, `operator()` and the previously mentioned *global kernel* `saxpy`:

* `set_arguments_size`: Sets the `size` of output parameters.
* `operator()`: The actual algorithm runs (similar to Gaudi).

In Allen, it is not recommended to use *dynamic memory allocations*. Therefore, types such as `std::vector` are "forbidden", and instead sizes of output arguments must be set in the `set_arguments_size` method of algorithms.

.. code-block:: c++

  #include "SAXPY_example.cuh"

  void saxpy::saxpy_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
  {
    set_size<dev_saxpy_output_t>(arguments, first<host_number_of_events_t>(arguments));
  }

To do that, one may use the following functions:

* `void set_size<T>(arguments, const size_t)`: Sets the size of *name* `T`. The `sizeof(T)` is implicit, so eg. `set_size<T>(10)` will actually allocate space for `10 * sizeof(T)`.
* `size_t size<T>(arguments)`: Gets the size of *name* `T`.
* `T* data<T>(arguments)`: Gets the pointer to the beginning of `T`.
* `T first<T>(arguments)`: Gets the first element of `T`.

Next, `operator()` should be defined:

.. code-block:: c++

  void saxpy::saxpy_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
  {
    global_function(saxpy)(
      dim3(1),
      property<block_dim_t>(), context)(arguments);
  }

In order to invoke host and global functions, wrapper methods `host_function` and `global_function` should be used. The syntax is as follows:

.. code-block:: c++

    host_function(<host_function_identifier>)(<parameters of function>)
    global_function(<global_function_identifier>)(<grid_size>, <block_size>, context)(<parameters of function>)

`global_function` wraps a function identifier, such as `saxpy`. The object it returns can be used to invoke a *global kernel* following a syntax that is similar to |cuda_kernel_guide|. It expects:

.. |cuda_kernel_guide| raw:: html

   <a href="<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels" target="_blank">CUDA's kernel invocation syntax</a>

* `grid_size`: Number of blocks of kernel invocation (passed as 3-dimensional object of type `dim3`).
* `block_size`: Number of threads in each block (passed as 3-dimensional object of type `dim3`).
* `stream`: Stream where to run.
* `parameters of function`: Parameters of the *global kernel* being invoked.

In this case, the kernel `saxpy` accepts only one parameter of type `Parameters`. The global_function and host_function wrappers automatically detect and transform `const ArgumentReferences<Parameters>&` into `Parameters`. Therefore, we can safely pass `arguments` to our kernel invocation.

Finally, the kernel is defined:

.. code-block:: c++

  /**
  * @brief SAXPY example algorithm
  * @detail Calculates for every event y = a*x + x, where x is the number of velo tracks in one event
  */
  __global__ void saxpy::saxpy(saxpy::Parameters parameters)
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

The kernel accepts a single parameter of type `saxpy::Parameters`. It is now possible to access all previously defined parameters by their *identifier*. Things to remember here:

* A parameter or property is accessed with its *identifier*.
* Parameters decays to *underlying type* (eg. formed from its *scope* and its *type*).
* Properties decay to *type*.
* If explicit access to the *underlying type* of parameters is required, `get()` can be used.
* One should not access `host` parameters inside a function to be executed on the `device`, and viceversa.

In other words, in the code above:

* `parameters.dev_number_of_events` decays to `const unsigned*`.
* `parameters.dev_atomics_velo` decays to `const unsigned*`.
* `parameters.dev_velo_track_hit_number` decays to `const unsigned*`.
* `parameters.dev_saxpy_output` decays to `float*`.
* `parameters.saxpy_scale_factor` decays to `float`, and has default value `2.f`.

.. _building newly defined algorithm:

Building with a newly defined algorithm
---------------------------------------

If a new algorithm was defined, or if an algorithm was removed, it is required to purge the project and trigger a rebuild from scratch.

This is due to a parsing of all Allen header files that occurs upon running cmake for the first time.

Separately, modifying existing algorithms in any way such as changing its parameters or its source code does not necessitate a purge.

How to access current event within algorithm
--------------------------------------------------

Typically, events are processed by independent blocks of execution. When that's the case, the invocation of the global function happens with as many blocks as events in the event list. Eg.

.. code-block:: c++

  global_function(kernel)(
    size<dev_event_list_t>(),
    property<block_dim_t>(), context)(arguments);

Then, in the kernel itself, in order to access the event under execution, the following idiom is used:

.. code-block:: c++

  __global__ void kernel(namespace::Parameters parameters) {
    const unsigned event_number = parameters.dev_event_list[blockIdx.x];

Configuring the algorithm in a sequence
---------------------------------------

The last thing remaining is to add the algorithm to a sequence, and run it.
:ref:`configure_sequence` explains how to configure the algorithms in an HLT1 sequence.
