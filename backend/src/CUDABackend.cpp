/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#include <BackendCommon.h>
#include <Logger.h>

/**
 * @brief Prints the memory consumption of the device.
 */
void Allen::print_device_memory_consumption()
{
  size_t free_byte;
  size_t total_byte;
  cudaCheck(cudaMemGetInfo(&free_byte, &total_byte));
  float free_percent = (float) free_byte / total_byte * 100;
  float used_percent = (float) (total_byte - free_byte) / total_byte * 100;
  verbose_cout << "GPU memory: " << free_percent << " percent free, " << used_percent << " percent used "
               << std::endl;
}

std::tuple<bool, std::string, unsigned> Allen::set_device(int cuda_device, size_t stream_id)
{
  int n_devices = 0;
  cudaDeviceProp device_properties;

  try {
    cudaCheck(cudaGetDeviceCount(&n_devices));

    debug_cout << "There are " << n_devices << " CUDA devices available\n";
    for (int cd = 0; cd < n_devices; ++cd) {
      cudaDeviceProp device_properties;
      cudaCheck(cudaGetDeviceProperties(&device_properties, cd));
      debug_cout << std::setw(3) << cd << " " << device_properties.name << "\n";
    }

    if (cuda_device >= n_devices) {
      error_cout << "Chosen device (" << cuda_device << ") is not available.\n";
      return {false, "", 0};
    }
    debug_cout << "\n";

    cudaCheck(cudaSetDevice(cuda_device));
    cudaCheck(cudaGetDeviceProperties(&device_properties, cuda_device));

    if (n_devices == 0) {
      error_cout << "Failed to select device " << cuda_device << "\n";
      return {false, "", 0};
    }
    else {
      debug_cout << "Stream " << stream_id << " selected cuda device " << cuda_device << ": "
                 << device_properties.name << "\n\n";
    }
  } catch (const std::invalid_argument& e) {
    error_cout << e.what() << std::endl;
    error_cout << "Stream " << stream_id << " failed to select cuda device " << cuda_device << "\n";
    return {false, "", 0};
  }

  if (device_properties.major == 7 && device_properties.minor == 5) {
    // Turing architecture benefits from setting up cache config to L1
    cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  }

  return {true, device_properties.name, device_properties.textureAlignment};
}

std::tuple<bool, int> Allen::get_device_id(const std::string& pci_bus_id)
{
  int device = 0;
  try {
    cudaCheck(cudaDeviceGetByPCIBusId(&device, pci_bus_id.c_str()));
  } catch (std::invalid_argument& a) {
    error_cout << "Failed to get device by PCI bus ID: " << pci_bus_id << "\n";
    return {false, 0};
  }
  return {true, device};
}
