/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

fn main() {
    // Only set up static linking if tensor_engine feature is enabled
    if std::env::var("CARGO_FEATURE_TENSOR_ENGINE").is_ok() {
        // Set up static linking for rdma-core
        // This emits link directives for libmlx5.a, libibverbs.a, librdma_util.a
        let _config = build_utils::setup_cpp_static_libs();

        // Link libfabric statically for EFA support.
        // Path is exported by rdmaxcel-sys via cargo:metadata=LIBFABRIC_A=...
        if let Ok(libfabric_a) = std::env::var("DEP_RDMAXCEL_LIBFABRIC_A") {
            println!("cargo:warning=Linking libfabric from {}", libfabric_a);
            println!("cargo:rustc-link-arg=-Wl,--whole-archive");
            println!("cargo:rustc-link-arg={}", libfabric_a);
            println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        }
    }
}
