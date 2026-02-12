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
        // Try DEP_RDMAXCEL_LIBFABRIC_A from rdmaxcel-sys metadata first,
        // then fall back to searching the rdmaxcel-sys build output.
        let libfabric_a = std::env::var("DEP_RDMAXCEL_LIBFABRIC_A").ok()
            .or_else(|| {
                let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
                let path = format!("{}/../rdmaxcel-sys/target/libfabric_build/libfabric-install/lib/libfabric.a", manifest);
                if std::path::Path::new(&path).exists() {
                    Some(std::fs::canonicalize(&path).ok()?.to_string_lossy().to_string())
                } else {
                    None
                }
            });

        if let Some(path) = &libfabric_a {
            println!("cargo:warning=Linking libfabric from {}", path);
            println!("cargo:rustc-link-arg=-Wl,--whole-archive");
            println!("cargo:rustc-link-arg={}", path);
            println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        }
    }
}
