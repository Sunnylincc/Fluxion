#include "kv_cache/block_allocator.h"

#include <algorithm>
#include <stdexcept>

namespace fluxion {

BlockAllocator::BlockAllocator(std::size_t total_blocks, std::size_t block_size_bytes)
    : total_blocks_(total_blocks), block_size_bytes_(block_size_bytes), used_bitmap_(total_blocks, false) {
  free_list_.reserve(total_blocks_);
  for (int i = static_cast<int>(total_blocks_) - 1; i >= 0; --i) {
    free_list_.push_back(i);
  }
}

std::vector<int> BlockAllocator::Allocate(const std::string& request_id, std::size_t num_blocks) {
  if (!CanAllocate(num_blocks)) {
    throw std::runtime_error("insufficient KV cache blocks");
  }

  auto& req_blocks = allocations_[request_id];
  for (std::size_t i = 0; i < num_blocks; ++i) {
    int block = free_list_.back();
    free_list_.pop_back();
    req_blocks.push_back(block);
    used_bitmap_[block] = true;
  }
  return req_blocks;
}

void BlockAllocator::Free(const std::string& request_id) {
  auto it = allocations_.find(request_id);
  if (it == allocations_.end()) {
    return;
  }

  for (int block : it->second) {
    if (block >= 0 && static_cast<std::size_t>(block) < used_bitmap_.size()) {
      used_bitmap_[block] = false;
    }
    free_list_.push_back(block);
  }
  allocations_.erase(it);
}

std::size_t BlockAllocator::LargestContiguousFreeRun() const {
  std::size_t best = 0;
  std::size_t cur = 0;
  for (bool used : used_bitmap_) {
    if (!used) {
      cur += 1;
      best = std::max(best, cur);
    } else {
      cur = 0;
    }
  }
  return best;
}

KVCacheStats BlockAllocator::GetStats() const {
  const std::size_t free_blocks = free_list_.size();
  const std::size_t used_blocks = total_blocks_ - free_blocks;
  const double utilization = total_blocks_ == 0 ? 0.0 : static_cast<double>(used_blocks) / static_cast<double>(total_blocks_);
  const std::size_t largest_run = LargestContiguousFreeRun();

  double external_fragmentation = 0.0;
  if (free_blocks > 0) {
    external_fragmentation = 1.0 - (static_cast<double>(largest_run) / static_cast<double>(free_blocks));
  }

  return KVCacheStats{total_blocks_, free_blocks, used_blocks, largest_run, utilization, external_fragmentation};
}

bool BlockAllocator::CanAllocate(std::size_t num_blocks) const {
  return free_list_.size() >= num_blocks;
}

}  // namespace fluxion
