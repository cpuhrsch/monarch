/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "rdmaxcel_efa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <unordered_map>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

// Debug macro - only prints when MONARCH_DEBUG_EFA=1
static inline bool efa_debug_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* env = getenv("MONARCH_DEBUG_EFA");
    enabled = (env && strcmp(env, "1") == 0) ? 1 : 0;
  }
  return enabled == 1;
}
#define EFA_DEBUG(...) do { if (efa_debug_enabled()) fprintf(stderr, __VA_ARGS__); } while(0)

// Helper macro for ep_create: call a libfabric function, destroy ep and return nullptr on failure.
#define EFA_EP_TRY(call, msg) \
  do { \
    int _ret = (call); \
    if (_ret != 0) { \
      print_fi_error(msg, _ret); \
      rdmaxcel_efa_ep_destroy(ep); \
      return nullptr; \
    } \
  } while(0)

// Info about a registered memory region
struct mr_info {
  struct fid_mr* mr;
  uintptr_t addr;
  size_t size;
  uint64_t provider_key; // key assigned by the provider (fi_mr_key)
  void* desc;            // cached fi_mr_desc() result
};

// Internal EFA endpoint structure
struct rdmaxcel_efa_ep {
  struct fi_info* fi;
  struct fid_fabric* fabric;
  struct fid_domain* domain;
  struct fid_ep* ep;
  struct fid_cq* cq;
  struct fid_av* av;
  std::unordered_map<uint64_t, mr_info> registered_mrs;
  uint64_t next_mr_key;
  char local_addr[256];
  size_t local_addr_len;
};

// Helper: Print libfabric error
static void print_fi_error(const char* msg, int err) {
  fprintf(stderr, "[EFA] %s: %s (%d)\n", msg, fi_strerror(-err), err);
}

// Initialize platform-specific settings for AWS EFA
static void init_efa_platform(void) {
  // Check fork-safe mode for EFA
  // For libfabric 1.13+, use FI_EFA_FORK_SAFE
  uint32_t libversion = fi_version();
  const char* fork_safe_var =
      (FI_MAJOR(libversion) > 1 ||
       (FI_MAJOR(libversion) == 1 && FI_MINOR(libversion) >= 13))
          ? "FI_EFA_FORK_SAFE"
          : "RDMAV_FORK_SAFE";

  // Only set if not already set by user
  if (!getenv(fork_safe_var)) {
    setenv(fork_safe_var, "0", 1);
  }
}

extern "C" {

int rdmaxcel_efa_available(void) {
  static int cached_result = -1;

  if (cached_result >= 0) {
    return cached_result;
  }

  struct fi_info* hints = fi_allocinfo();
  if (!hints) {
    cached_result = 0;
    return 0;
  }

  hints->fabric_attr->prov_name = strdup("efa");
  hints->ep_attr->type = FI_EP_RDM;
  hints->caps = FI_RMA | FI_READ | FI_WRITE | FI_MSG | FI_SEND | FI_RECV;

  struct fi_info* info = nullptr;
  int ret = fi_getinfo(FI_VERSION(1, 22), nullptr, nullptr, FI_SOURCE, hints, &info);
  fi_freeinfo(hints);

  if (ret == 0 && info) {
    // Check if we found actual EFA devices (not just TCP fallback)
    bool found_efa = false;
    for (struct fi_info* cur = info; cur; cur = cur->next) {
      if (cur->domain_attr && cur->domain_attr->name &&
          strstr(cur->domain_attr->name, "rdma")) {
        found_efa = true;
        EFA_DEBUG("[EFA] Found device: %s\n", cur->domain_attr->name);
        break;
      }
    }
    fi_freeinfo(info);
    cached_result = found_efa ? 1 : 0;
    return cached_result;
  }

  cached_result = 0;
  return 0;
}

rdmaxcel_efa_ep_t* rdmaxcel_efa_ep_create(const char* provider) {
  // Initialize platform settings on first use
  static bool platform_initialized = false;
  if (!platform_initialized) {
    init_efa_platform();
    platform_initialized = true;
  }

  // Use new instead of calloc to properly construct C++ members (std::unordered_map)
  rdmaxcel_efa_ep_t* ep = new (std::nothrow) rdmaxcel_efa_ep_t();
  if (!ep) {
    return nullptr;
  }

  ep->next_mr_key = 1;
  ep->local_addr_len = sizeof(ep->local_addr);

  // Set up fabric hints
  struct fi_info* hints = fi_allocinfo();
  if (!hints) {
    rdmaxcel_efa_ep_destroy(ep);
    return nullptr;
  }

  hints->fabric_attr->prov_name = strdup(provider ? provider : "efa");
  hints->ep_attr->type = FI_EP_RDM;
  hints->caps =
      FI_RMA | FI_READ | FI_WRITE | FI_MSG | FI_SEND | FI_RECV | FI_TAGGED;
  hints->mode = FI_CONTEXT;
  hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
  hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;

  int ret = fi_getinfo(FI_VERSION(1, 22), nullptr, nullptr, FI_SOURCE, hints, &ep->fi);
  fi_freeinfo(hints);

  if (ret != 0 || !ep->fi) {
    print_fi_error("fi_getinfo failed", ret);
    rdmaxcel_efa_ep_destroy(ep);
    return nullptr;
  }

  // Create fabric, domain, CQ, AV, endpoint — on failure, destroy cleans up
  // everything allocated so far since fields are zero-initialized.
  EFA_EP_TRY(fi_fabric(ep->fi->fabric_attr, &ep->fabric, nullptr), "fi_fabric failed");
  EFA_EP_TRY(fi_domain(ep->fabric, ep->fi, &ep->domain, nullptr), "fi_domain failed");

  struct fi_cq_attr cq_attr = {};
  cq_attr.size = 8192;
  cq_attr.format = FI_CQ_FORMAT_TAGGED;
  cq_attr.wait_obj = FI_WAIT_NONE;
  EFA_EP_TRY(fi_cq_open(ep->domain, &cq_attr, &ep->cq, nullptr), "fi_cq_open failed");

  struct fi_av_attr av_attr = {};
  av_attr.type = FI_AV_MAP;
  av_attr.count = 16;
  EFA_EP_TRY(fi_av_open(ep->domain, &av_attr, &ep->av, nullptr), "fi_av_open failed");

  EFA_EP_TRY(fi_endpoint(ep->domain, ep->fi, &ep->ep, nullptr), "fi_endpoint failed");
  EFA_EP_TRY(fi_ep_bind(ep->ep, &ep->cq->fid, FI_TRANSMIT | FI_RECV), "fi_ep_bind CQ failed");
  EFA_EP_TRY(fi_ep_bind(ep->ep, &ep->av->fid, 0), "fi_ep_bind AV failed");
  EFA_EP_TRY(fi_enable(ep->ep), "fi_enable failed");

  // Get local address for later connection
  ret = fi_getname(&ep->ep->fid, ep->local_addr, &ep->local_addr_len);
  if (ret != 0) {
    print_fi_error("fi_getname failed", ret);
  }

  return ep;
}

void rdmaxcel_efa_ep_destroy(rdmaxcel_efa_ep_t* ep) {
  if (!ep) {
    return;
  }

  // Deregister all MRs
  for (auto& pair : ep->registered_mrs) {
    if (pair.second.mr) {
      fi_close(&pair.second.mr->fid);
    }
  }
  ep->registered_mrs.clear();

  // Close resources in reverse order
  if (ep->ep) {
    fi_close(&ep->ep->fid);
  }
  if (ep->av) {
    fi_close(&ep->av->fid);
  }
  if (ep->cq) {
    fi_close(&ep->cq->fid);
  }
  if (ep->domain) {
    fi_close(&ep->domain->fid);
  }
  if (ep->fabric) {
    fi_close(&ep->fabric->fid);
  }
  if (ep->fi) {
    fi_freeinfo(ep->fi);
  }

  delete ep;
}

int rdmaxcel_efa_get_local_addr(
    rdmaxcel_efa_ep_t* ep,
    void* addr_out,
    size_t* addr_len) {
  if (!ep || !addr_out || !addr_len) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  if (*addr_len < ep->local_addr_len) {
    *addr_len = ep->local_addr_len;
    return EFA_ERROR_INVALID_PARAMS;
  }

  memcpy(addr_out, ep->local_addr, ep->local_addr_len);
  *addr_len = ep->local_addr_len;
  return EFA_SUCCESS;
}

int rdmaxcel_efa_insert_peer_addr(
    rdmaxcel_efa_ep_t* ep,
    const void* peer_addr,
    size_t addr_len,
    uint64_t* fi_addr_out) {
  if (!ep || !peer_addr || addr_len == 0 || !fi_addr_out) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  fi_addr_t fi_addr;
  int ret = fi_av_insert(ep->av, peer_addr, 1, &fi_addr, 0, nullptr);
  if (ret != 1) {
    print_fi_error("fi_av_insert failed", ret);
    return EFA_ERROR_AV_FAILED;
  }

  *fi_addr_out = static_cast<uint64_t>(fi_addr);
  EFA_DEBUG("[EFA] insert_peer_addr: fi_addr=%lu\n", (unsigned long)fi_addr);
  return EFA_SUCCESS;
}

int rdmaxcel_efa_register_mr(
    rdmaxcel_efa_ep_t* ep,
    void* addr,
    size_t size,
    uint64_t* key_out) {
  if (!ep || !addr || size == 0 || !key_out) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  // Pre-fault all pages before registration to avoid hangs in fi_mr_reg
  // Touch every 4KB page to ensure they're backed by physical memory
  EFA_DEBUG("[EFA] Pre-faulting %zu bytes (%zu MB) before registration...\n",
          size, size / (1024 * 1024));
  volatile char* ptr = static_cast<volatile char*>(addr);
  const size_t page_size = 4096;
  for (size_t i = 0; i < size; i += page_size) {
    // Read to fault in the page
    (void)ptr[i];
  }
  EFA_DEBUG("[EFA] Pre-faulting complete, calling fi_mr_reg...\n");

  struct fid_mr* mr = nullptr;
  uint64_t access =
      FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV;

  int ret = fi_mr_reg(
      ep->domain, addr, size, access, 0, ep->next_mr_key, 0, &mr, nullptr);
  if (ret != 0) {
    print_fi_error("fi_mr_reg failed", ret);
    return EFA_ERROR_MR_FAILED;
  }

  EFA_DEBUG("[EFA] fi_mr_reg succeeded\n");
  *key_out = fi_mr_key(mr);
  uint64_t internal_id = ep->next_mr_key;
  void* desc = fi_mr_desc(mr);
  ep->registered_mrs[internal_id] = {mr, reinterpret_cast<uintptr_t>(addr), size, *key_out, desc};
  ep->next_mr_key++;

  EFA_DEBUG("[EFA] register_mr: addr=%p, size=%zu, provider_key=%lu, internal_id=%lu, total_mrs=%zu\n",
          addr, size, (unsigned long)*key_out, (unsigned long)internal_id, ep->registered_mrs.size());

  return EFA_SUCCESS;
}

int rdmaxcel_efa_deregister_mr(rdmaxcel_efa_ep_t* ep, uint64_t key) {
  if (!ep) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  // Find MR by provider key (the Rust layer passes fi_mr_key)
  for (auto it = ep->registered_mrs.begin(); it != ep->registered_mrs.end(); ++it) {
    if (it->second.provider_key == key) {
      EFA_DEBUG("[EFA] deregister_mr: provider_key=%lu, addr=0x%lx, remaining_mrs=%zu\n",
              (unsigned long)key, (unsigned long)it->second.addr, ep->registered_mrs.size() - 1);
      fi_close(&it->second.mr->fid);
      ep->registered_mrs.erase(it);
      return EFA_SUCCESS;
    }
  }

  EFA_DEBUG("[EFA] deregister_mr: key=%lu NOT FOUND in %zu MRs\n",
          (unsigned long)key, ep->registered_mrs.size());
  return EFA_ERROR_INVALID_PARAMS;
}

int rdmaxcel_efa_write(
    rdmaxcel_efa_ep_t* ep,
    void* local_addr,
    size_t size,
    uint64_t remote_addr,
    uint64_t remote_key,
    uint64_t peer) {
  if (!ep || !local_addr || size == 0) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  // Find local MR for descriptor - must match the address being written from
  void* desc = nullptr;
  uintptr_t addr_val = reinterpret_cast<uintptr_t>(local_addr);
  for (auto& pair : ep->registered_mrs) {
    const mr_info& info = pair.second;
    if (addr_val >= info.addr && addr_val < info.addr + info.size) {
      desc = info.desc;
      break;
    }
  }
  if (!desc) {
    EFA_DEBUG("[EFA] write: no MR found covering local_addr=%p (have %zu MRs)\n",
            local_addr, ep->registered_mrs.size());
    if (efa_debug_enabled()) {
      for (auto& pair : ep->registered_mrs) {
        fprintf(stderr, "  MR key=%lu addr=0x%lx size=%zu\n",
                (unsigned long)pair.first, (unsigned long)pair.second.addr, pair.second.size);
      }
    }
    return EFA_ERROR_INVALID_PARAMS;
  }

  fi_addr_t fi_peer = static_cast<fi_addr_t>(peer);
  EFA_DEBUG("[EFA] write: local_addr=%p, size=%zu, remote_addr=0x%lx, remote_key=%lu, peer=%lu, num_mrs=%zu\n",
          local_addr, size, (unsigned long)remote_addr, (unsigned long)remote_key,
          (unsigned long)fi_peer, ep->registered_mrs.size());

  // Post write with retry on EAGAIN
  for (int retries = 0; retries < 100000; retries++) {
    ssize_t ret = fi_write(
        ep->ep,
        local_addr,
        size,
        desc,
        fi_peer,
        remote_addr,
        remote_key,
        nullptr);

    if (ret == 0) {
      EFA_DEBUG("[EFA] write: fi_write succeeded\n");
      return EFA_SUCCESS;
    } else if (ret == -FI_EAGAIN) {
      // Resource temporarily unavailable, poll CQ and retry
      struct fi_cq_tagged_entry entry;
      fi_cq_read(ep->cq, &entry, 1);
      continue;
    } else {
      EFA_DEBUG("[EFA] write: fi_write FAILED with ret=%zd (%s)\n", ret, fi_strerror(-ret));
      print_fi_error("fi_write failed", static_cast<int>(ret));
      return EFA_ERROR_WRITE_FAILED;
    }
  }

  return EFA_ERROR_WRITE_FAILED;
}

int rdmaxcel_efa_read(
    rdmaxcel_efa_ep_t* ep,
    void* local_addr,
    size_t size,
    uint64_t remote_addr,
    uint64_t remote_key,
    uint64_t peer) {
  if (!ep || !local_addr || size == 0) {
    EFA_DEBUG("[EFA] read: invalid params (ep=%p, local_addr=%p, size=%zu)\n", ep, local_addr, size);
    return EFA_ERROR_INVALID_PARAMS;
  }

  // Find local MR for descriptor - must match the address being read into
  void* desc = nullptr;
  uintptr_t addr_val = reinterpret_cast<uintptr_t>(local_addr);
  for (auto& pair : ep->registered_mrs) {
    const mr_info& info = pair.second;
    if (addr_val >= info.addr && addr_val < info.addr + info.size) {
      desc = info.desc;
      break;
    }
  }

  if (!desc) {
    EFA_DEBUG("[EFA] read: no MR found covering local_addr=%p\n", local_addr);
    return EFA_ERROR_MR_FAILED;
  }

  fi_addr_t fi_peer = static_cast<fi_addr_t>(peer);
  EFA_DEBUG("[EFA] read: posting fi_read local=%p size=%zu remote=0x%lx key=%lu peer=%lu\n",
          local_addr, size, remote_addr, remote_key, (unsigned long)fi_peer);

  // Post read with retry on EAGAIN
  for (int retries = 0; retries < 100000; retries++) {
    ssize_t ret = fi_read(
        ep->ep,
        local_addr,
        size,
        desc,
        fi_peer,
        remote_addr,
        remote_key,
        nullptr);

    if (ret == 0) {
      EFA_DEBUG("[EFA] read: fi_read posted successfully\n");
      return EFA_SUCCESS;
    } else if (ret == -FI_EAGAIN) {
      struct fi_cq_tagged_entry entry;
      fi_cq_read(ep->cq, &entry, 1);
      continue;
    } else {
      print_fi_error("fi_read failed", static_cast<int>(ret));
      return EFA_ERROR_READ_FAILED;
    }
  }

  EFA_DEBUG("[EFA] read: exhausted retries\n");
  return EFA_ERROR_READ_FAILED;
}

int rdmaxcel_efa_poll_cq(rdmaxcel_efa_ep_t* ep, int timeout_ms) {
  if (!ep) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  struct fi_cq_tagged_entry entry;
  ssize_t ret;

  if (timeout_ms == 0) {
    // Non-blocking poll
    ret = fi_cq_read(ep->cq, &entry, 1);
  } else if (timeout_ms < 0) {
    // Blocking wait - poll in a loop
    while (true) {
      ret = fi_cq_read(ep->cq, &entry, 1);
      if (ret > 0 || (ret < 0 && ret != -FI_EAGAIN)) {
        break;
      }
    }
  } else {
    // Timed wait - busy poll with timeout check
    // (fi_cq_sread requires FI_WAIT_FD which we don't use)
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    int64_t timeout_ns = static_cast<int64_t>(timeout_ms) * 1000000LL;

    while (true) {
      ret = fi_cq_read(ep->cq, &entry, 1);
      if (ret > 0 || (ret < 0 && ret != -FI_EAGAIN)) {
        break;
      }

      // Check if timeout has elapsed
      clock_gettime(CLOCK_MONOTONIC, &current_time);
      int64_t elapsed_ns = (current_time.tv_sec - start_time.tv_sec) * 1000000000LL +
                           (current_time.tv_nsec - start_time.tv_nsec);
      if (elapsed_ns >= timeout_ns) {
        ret = -FI_EAGAIN;  // Treat as no completions available
        break;
      }
    }
  }

  if (ret > 0) {
    return static_cast<int>(ret);
  } else if (ret == -FI_EAGAIN) {
    return 0; // No completions available
  } else if (ret == -FI_ETIMEDOUT) {
    return EFA_ERROR_TIMEOUT;
  } else if (ret == -FI_EAVAIL) {
    // Error available - read it
    struct fi_cq_err_entry err_entry;
    ssize_t err_ret = fi_cq_readerr(ep->cq, &err_entry, 0);
    if (err_ret > 0) {
      fprintf(stderr, "[EFA] CQ error: %s (prov_errno=%d)\n",
              fi_cq_strerror(ep->cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0),
              err_entry.prov_errno);
    }
    return EFA_ERROR_POLL_FAILED;
  } else {
    fprintf(stderr, "[EFA] poll_cq failed with error: %s (%zd)\n", fi_strerror(-ret), ret);
    return EFA_ERROR_POLL_FAILED;
  }
}

int rdmaxcel_efa_tsend(rdmaxcel_efa_ep_t* ep, uint64_t tag, uint64_t peer) {
  if (!ep) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  // Send a minimal tagged message (1 byte) as completion notification
  static char completion_byte = 1;

  fi_addr_t fi_peer = static_cast<fi_addr_t>(peer);
  EFA_DEBUG("[EFA] tsend: sending with tag=%lu peer=%lu\n", tag, (unsigned long)fi_peer);
  for (int retries = 0; retries < 100000; retries++) {
    ssize_t ret = fi_tsend(
        ep->ep,
        &completion_byte,
        sizeof(completion_byte),
        nullptr,  // No descriptor needed for small message
        fi_peer,
        tag,
        nullptr);  // No context

    if (ret == 0) {
      EFA_DEBUG("[EFA] tsend: posted successfully\n");
      return EFA_SUCCESS;
    } else if (ret == -FI_EAGAIN) {
      // Resource temporarily unavailable, poll CQ and retry
      struct fi_cq_tagged_entry entry;
      fi_cq_read(ep->cq, &entry, 1);
      continue;
    } else {
      print_fi_error("fi_tsend failed", static_cast<int>(ret));
      return EFA_ERROR_WRITE_FAILED;
    }
  }

  return EFA_ERROR_WRITE_FAILED;
}

int rdmaxcel_efa_trecv(rdmaxcel_efa_ep_t* ep, uint64_t tag) {
  if (!ep) {
    return EFA_ERROR_INVALID_PARAMS;
  }

  // Post a tagged receive for completion notification
  // Use exact tag matching to avoid confusion with concurrent transfers
  static char completion_byte = 0;

  EFA_DEBUG("[EFA] trecv: posting with tag=%lu\n", tag);
  for (int retries = 0; retries < 100000; retries++) {
    ssize_t ret = fi_trecv(
        ep->ep,
        &completion_byte,
        sizeof(completion_byte),
        nullptr,  // No descriptor needed for small message
        FI_ADDR_UNSPEC,  // Accept from any source
        tag,
        0,  // No ignore mask - exact tag match
        nullptr);  // No context

    if (ret == 0) {
      EFA_DEBUG("[EFA] trecv: posted successfully\n");
      return EFA_SUCCESS;
    } else if (ret == -FI_EAGAIN) {
      // Resource temporarily unavailable, poll CQ and retry
      struct fi_cq_tagged_entry entry;
      fi_cq_read(ep->cq, &entry, 1);
      continue;
    } else {
      print_fi_error("fi_trecv failed", static_cast<int>(ret));
      return EFA_ERROR_READ_FAILED;
    }
  }

  return EFA_ERROR_READ_FAILED;
}

const char* rdmaxcel_efa_error_string(int error_code) {
  switch (error_code) {
    case EFA_SUCCESS:
      return "Success";
    case EFA_ERROR_NOT_AVAILABLE:
      return "EFA not available (not compiled with libfabric support)";
    case EFA_ERROR_INIT_FAILED:
      return "EFA initialization failed";
    case EFA_ERROR_FABRIC_FAILED:
      return "Failed to create fabric";
    case EFA_ERROR_DOMAIN_FAILED:
      return "Failed to create domain";
    case EFA_ERROR_EP_FAILED:
      return "Failed to create endpoint";
    case EFA_ERROR_CQ_FAILED:
      return "Failed to create completion queue";
    case EFA_ERROR_AV_FAILED:
      return "Failed to create or insert into address vector";
    case EFA_ERROR_MR_FAILED:
      return "Failed to register memory region";
    case EFA_ERROR_CONNECT_FAILED:
      return "Failed to connect (no peer address)";
    case EFA_ERROR_WRITE_FAILED:
      return "RDMA write operation failed";
    case EFA_ERROR_READ_FAILED:
      return "RDMA read operation failed";
    case EFA_ERROR_POLL_FAILED:
      return "Completion queue poll failed";
    case EFA_ERROR_TIMEOUT:
      return "Operation timed out";
    case EFA_ERROR_INVALID_PARAMS:
      return "Invalid parameters provided";
    default:
      return "Unknown EFA error";
  }
}

} // extern "C"
