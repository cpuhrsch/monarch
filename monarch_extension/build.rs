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
        // Built from source by rdmaxcel-sys/build.rs — must exist if rdmaxcel-sys compiled.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let libfabric_a = std::fs::canonicalize(
            format!("{}/../rdmaxcel-sys/target/libfabric_build/libfabric-install/lib/libfabric.a", manifest)
        ).expect("libfabric.a not found — rdmaxcel-sys build may have failed");
        println!("cargo:warning=Linking libfabric from {}", libfabric_a.display());
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg={}", libfabric_a.display());
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    }
}
