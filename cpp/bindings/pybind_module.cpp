#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_cache/block_allocator.h"

namespace py = pybind11;

PYBIND11_MODULE(fluxion_cpp, m) {
  py::class_<fluxion::KVCacheStats>(m, "KVCacheStats")
      .def_readonly("total_blocks", &fluxion::KVCacheStats::total_blocks)
      .def_readonly("free_blocks", &fluxion::KVCacheStats::free_blocks)
      .def_readonly("used_blocks", &fluxion::KVCacheStats::used_blocks)
      .def_readonly("largest_contiguous_free_run", &fluxion::KVCacheStats::largest_contiguous_free_run)
      .def_readonly("utilization", &fluxion::KVCacheStats::utilization)
      .def_readonly("external_fragmentation", &fluxion::KVCacheStats::external_fragmentation);

  py::class_<fluxion::BlockAllocator>(m, "BlockAllocator")
      .def(py::init<std::size_t, std::size_t>())
      .def("allocate", &fluxion::BlockAllocator::Allocate)
      .def("free", &fluxion::BlockAllocator::Free)
      .def("stats", &fluxion::BlockAllocator::GetStats)
      .def("can_allocate", &fluxion::BlockAllocator::CanAllocate);
}
