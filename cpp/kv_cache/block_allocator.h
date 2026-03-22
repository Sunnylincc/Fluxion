#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace fluxion {

struct KVCacheStats {
  std::size_t total_blocks;
  std::size_t free_blocks;
  std::size_t used_blocks;
  std::size_t largest_contiguous_free_run;
  double utilization;
  double external_fragmentation;
};

class BlockAllocator {
 public:
  BlockAllocator(std::size_t total_blocks, std::size_t block_size_bytes);

  std::vector<int> Allocate(const std::string& request_id, std::size_t num_blocks);
  void Free(const std::string& request_id);
  KVCacheStats GetStats() const;
  bool CanAllocate(std::size_t num_blocks) const;

 private:
  std::size_t LargestContiguousFreeRun() const;

  std::size_t total_blocks_;
  std::size_t block_size_bytes_;
  std::vector<int> free_list_;
  std::vector<bool> used_bitmap_;
  std::unordered_map<std::string, std::vector<int>> allocations_;
};

}  // namespace fluxion
