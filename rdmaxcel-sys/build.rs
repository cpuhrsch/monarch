/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::Path;
use std::path::PathBuf;

const LIBFABRIC_REPO: &str = "https://github.com/aws/libfabric";
const LIBFABRIC_TAG: &str = "v1.22.0amzn4.0";

/// Run a command, returning Ok(()) on success or Err with stderr on failure.
#[cfg(not(target_os = "macos"))]
fn run_cmd(cmd: &mut std::process::Command) -> Result<(), String> {
    let output = cmd.output().map_err(|e| format!("failed to execute: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

/// Build libfabric from source and return (include_dir, lib_dir).
/// Caches the build at rdmaxcel-sys/target/libfabric_build/.
/// Returns None if the build fails (EFA will be disabled).
#[cfg(not(target_os = "macos"))]
fn build_libfabric() -> Option<(String, String)> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let base = format!("{}/target/libfabric_build", manifest_dir);
    let src = format!("{}/libfabric", base);
    let build_dir = format!("{}/libfabric-build", base);
    let install = format!("{}/libfabric-install", base);
    let lib_a = format!("{}/lib/libfabric.a", install);

    // Return cached build if available
    if Path::new(&lib_a).exists() {
        return Some((format!("{}/include", install), format!("{}/lib", install)));
    }

    std::fs::create_dir_all(&base).expect("Failed to create libfabric build dir");

    // Clone source if needed
    if !Path::new(&format!("{}/configure.ac", src)).exists() {
        let tag = env::var("MONARCH_LIBFABRIC_TAG").unwrap_or(LIBFABRIC_TAG.to_string());
        println!("cargo:warning=Cloning libfabric from {LIBFABRIC_REPO} (tag {tag})");
        run_cmd(std::process::Command::new("git").args([
            "clone", "--depth=1", "--branch", &tag, LIBFABRIC_REPO, &src,
        ])).ok()?;
    }

    // Generate configure script
    if !Path::new(&format!("{}/configure", src)).exists() {
        println!("cargo:warning=Running libfabric autogen.sh...");
        run_cmd(std::process::Command::new("bash").arg("autogen.sh").current_dir(&src)).ok()?;
    }

    // Configure
    std::fs::create_dir_all(&build_dir).ok()?;
    std::fs::create_dir_all(&install).ok()?;
    println!("cargo:warning=Configuring libfabric...");
    run_cmd(
        std::process::Command::new(format!("{}/configure", src))
            .args([
                &format!("--prefix={}", install),
                "--enable-static", "--disable-shared", "--with-pic",
                "--enable-efa=yes",
                // Disable unneeded providers to speed up build
                "--enable-tcp=no", "--enable-udp=no", "--enable-sockets=no",
                "--enable-rxm=no", "--enable-mrail=no", "--enable-rxd=no",
                "--enable-shm=no", "--enable-rstream=no", "--enable-perf=no",
                "--enable-hook_debug=no", "--enable-dmabuf_peer_mem=no",
            ])
            // Disable symbol versioning so rust-lld can link the static library.
            // HAVE_SYMVER_SUPPORT controls the .symver asm directives in ofi_abi.h.
            .env("CFLAGS", "-fPIC -O2")
            .current_dir(&build_dir),
    ).ok()?;

    // Patch config.h to:
    // 1. Disable symbol versioning so rust-lld can link without version scripts
    // 2. Force all providers as builtin (not DSO) so they auto-register at init
    let config_h = format!("{}/config.h", build_dir);
    if Path::new(&config_h).exists() {
        let content = std::fs::read_to_string(&config_h).unwrap_or_default();
        let patched = content
            .replace("#define HAVE_SYMVER_SUPPORT 1", "#define HAVE_SYMVER_SUPPORT 0")
            .replace("#define HAVE_EFA_DL 1", "#define HAVE_EFA_DL 0")
            .replace("#define HAVE_VERBS_DL 1", "#define HAVE_VERBS_DL 0")
            .replace("#define HAVE_COLL_DL 1", "#define HAVE_COLL_DL 0")
            .replace("#define HAVE_HOOK_HMEM_DL 1", "#define HAVE_HOOK_HMEM_DL 0")
            .replace("#define HAVE_TRACE_DL 1", "#define HAVE_TRACE_DL 0")
            .replace("#define HAVE_OPX_DL 1", "#define HAVE_OPX_DL 0")
            .replace("#define HAVE_USNIC_DL 1", "#define HAVE_USNIC_DL 0")
            .replace("#define HAVE_PSM3_DL 1", "#define HAVE_PSM3_DL 0");
        std::fs::write(&config_h, patched).ok();
        println!("cargo:warning=Patched config.h: disabled SYMVER + forced all providers builtin");
    }

    // Build and install
    println!("cargo:warning=Building libfabric...");
    let nproc = std::process::Command::new("nproc").output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(4);
    run_cmd(std::process::Command::new("make").args(["-j", &nproc.to_string()]).current_dir(&build_dir)).ok()?;
    run_cmd(std::process::Command::new("make").arg("install").current_dir(&build_dir)).ok()?;
    println!("cargo:warning=libfabric build complete");

    Path::new(&lib_a).exists().then(|| {
        (format!("{}/include", install), format!("{}/lib", install))
    })
}

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    // Get rdma-core config from cpp_static_libs (includes are used, links emitted by monarch_extension)
    let cpp_static_libs_config = build_utils::CppStaticLibsConfig::from_env();
    let rdma_include = &cpp_static_libs_config.rdma_include_dir;

    // Link against dl for dynamic loading
    println!("cargo:rustc-link-lib=dl");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/rdmaxcel.h");
    println!("cargo:rerun-if-changed=src/rdmaxcel.c");
    println!("cargo:rerun-if-changed=src/rdmaxcel.cpp");
    println!("cargo:rerun-if-changed=src/driver_api.h");
    println!("cargo:rerun-if-changed=src/driver_api.cpp");

    // Validate CUDA installation and get CUDA home path
    let cuda_home = match build_utils::validate_cuda_installation() {
        Ok(home) => home,
        Err(_) => {
            build_utils::print_cuda_error_help();
            std::process::exit(1);
        }
    };

    // Get the directory of the current crate
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        // For buck2 run, we know the package is in fbcode/monarch/rdmaxcel-sys
        // Get the fbsource directory from the current directory path
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        let current_path = current_dir.to_string_lossy();

        // Find the fbsource part of the path
        if let Some(fbsource_pos) = current_path.find("fbsource") {
            let fbsource_path = &current_path[..fbsource_pos + "fbsource".len()];
            format!("{}/fbcode/monarch/rdmaxcel-sys", fbsource_path)
        } else {
            // If we can't find fbsource in the path, just use the current directory
            format!("{}/src", current_dir.to_string_lossy())
        }
    });

    // Create the absolute path to the header file
    let header_path = format!("{}/src/rdmaxcel.h", manifest_dir);
    let efa_header_path = format!("{}/src/rdmaxcel_efa.h", manifest_dir);

    // Check if the header file exists
    if !Path::new(&header_path).exists() {
        panic!("Header file not found at {}", header_path);
    }

    // Start building the bindgen configuration
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header(&header_path)
        .header(&efa_header_path)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow the specified functions, types, and variables
        .allowlist_function("ibv_.*")
        .allowlist_function("mlx5dv_.*")
        .allowlist_function("mlx5_wqe_.*")
        .allowlist_function("create_qp")
        .allowlist_function("create_mlx5dv_.*")
        .allowlist_function("register_cuda_memory")
        .allowlist_function("db_ring")
        .allowlist_function("cqe_poll")
        .allowlist_function("send_wqe")
        .allowlist_function("recv_wqe")
        .allowlist_function("launch_db_ring")
        .allowlist_function("launch_cqe_poll")
        .allowlist_function("launch_send_wqe")
        .allowlist_function("launch_recv_wqe")
        .allowlist_function("rdma_get_active_segment_count")
        .allowlist_function("rdma_get_all_segment_info")
        .allowlist_function("register_segments")
        .allowlist_function("deregister_segments")
        .allowlist_function("rdmaxcel_cu.*")
        .allowlist_function("get_cuda_pci_address_from_ptr")
        .allowlist_function("rdmaxcel_print_device_info")
        .allowlist_function("rdmaxcel_error_string")
        .allowlist_function("rdmaxcel_qp_.*")
        .allowlist_function("rdmaxcel_register_segment_scanner")
        .allowlist_function("poll_cq_with_cache")
        .allowlist_function("completion_cache_.*")
        // EFA functions
        .allowlist_function("rdmaxcel_efa_.*")
        .allowlist_type("ibv_.*")
        .allowlist_type("mlx5dv_.*")
        .allowlist_type("mlx5_wqe_.*")
        .allowlist_type("cqe_poll_result_t")
        .allowlist_type("wqe_params_t")
        .allowlist_type("cqe_poll_params_t")
        .allowlist_type("rdma_segment_info_t")
        .allowlist_type("rdmaxcel_scanned_segment_t")
        .allowlist_type("rdmaxcel_qp_t")
        .allowlist_type("rdmaxcel_qp")
        .allowlist_type("completion_cache_t")
        .allowlist_type("completion_cache")
        .allowlist_type("poll_context_t")
        .allowlist_type("poll_context")
        .allowlist_type("rdmaxcel_segment_scanner_fn")
        // EFA types
        .allowlist_type("efa_error_code_t")
        .allowlist_type("rdmaxcel_efa_ep_t")
        .allowlist_type("rdmaxcel_efa_ep")
        .allowlist_var("MLX5_.*")
        .allowlist_var("IBV_.*")
        .allowlist_var("EFA_.*")
        // Block specific types that are manually defined in lib.rs
        .blocklist_type("ibv_wc")
        .blocklist_type("mlx5_wqe_ctrl_seg")
        // Apply the same bindgen flags as in the BUCK file
        .bitfield_enum("ibv_access_flags")
        .bitfield_enum("ibv_qp_attr_mask")
        .bitfield_enum("ibv_wc_flags")
        .bitfield_enum("ibv_send_flags")
        .bitfield_enum("ibv_port_cap_flags")
        .constified_enum_module("ibv_qp_type")
        .constified_enum_module("ibv_qp_state")
        .constified_enum_module("ibv_port_state")
        .constified_enum_module("ibv_wc_opcode")
        .constified_enum_module("ibv_wr_opcode")
        .constified_enum_module("ibv_wc_status")
        .derive_default(true)
        .prepend_enum_name(false);

    // Add CUDA include path (we already validated it exists)
    let cuda_include_path = format!("{}/include", cuda_home);
    println!("cargo:rustc-env=CUDA_INCLUDE_PATH={}", cuda_include_path);
    builder = builder.clang_arg(format!("-I{}", cuda_include_path));

    // Add rdma-core include path from nccl-static-sys
    builder = builder.clang_arg(format!("-I{}", rdma_include));

    // Include headers and libs from the active environment.
    let python_config = match build_utils::python_env_dirs_with_interpreter("python3") {
        Ok(config) => config,
        Err(_) => {
            eprintln!("Warning: Failed to get Python environment directories");
            build_utils::PythonConfig {
                include_dir: None,
                lib_dir: None,
            }
        }
    };

    if let Some(include_dir) = &python_config.include_dir {
        builder = builder.clang_arg(format!("-I{}", include_dir));
    }
    if let Some(lib_dir) = &python_config.lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:metadata=LIB_PATH={}", lib_dir);
    }

    // Get CUDA library directory and emit link directives
    let cuda_lib_dir = build_utils::get_cuda_lib_dir();
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    // Note: libcuda is now loaded dynamically via dlopen in driver_api.cpp
    // Link cudart statically (CUDA Runtime API)
    println!("cargo:rustc-link-lib=static=cudart_static");
    // cudart_static requires linking against librt and libpthread
    println!("cargo:rustc-link-lib=rt");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");

    // Note: We no longer link against libtorch/c10 since segment scanning
    // is now done via a callback registered from the extension crate.

    // Generate bindings
    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    match env::var("OUT_DIR") {
        Ok(out_dir) => {
            // Export OUT_DIR so dependent crates can find our compiled libraries
            println!("cargo:out_dir={}", out_dir);

            let out_path = PathBuf::from(&out_dir);
            match bindings.write_to_file(out_path.join("bindings.rs")) {
                Ok(_) => {
                    println!("cargo:rustc-cfg=cargo");
                    println!("cargo:rustc-check-cfg=cfg(cargo)");
                }
                Err(e) => eprintln!("Warning: Couldn't write bindings: {}", e),
            }

            // Compile the C source file
            let c_source_path = format!("{}/src/rdmaxcel.c", manifest_dir);
            if Path::new(&c_source_path).exists() {
                let mut build = cc::Build::new();
                build
                    .file(&c_source_path)
                    .include(format!("{}/src", manifest_dir))
                    .include(rdma_include)
                    .flag("-fPIC");

                // Add CUDA include paths - reuse the paths we already found for bindgen
                build.include(&cuda_include_path);

                build.compile("rdmaxcel");
            } else {
                panic!("C source file not found at {}", c_source_path);
            }

            // Compile the C++ source file
            let cpp_source_path = format!("{}/src/rdmaxcel.cpp", manifest_dir);
            let driver_api_cpp_path = format!("{}/src/driver_api.cpp", manifest_dir);
            if Path::new(&cpp_source_path).exists() && Path::new(&driver_api_cpp_path).exists() {
                let mut cpp_build = cc::Build::new();
                cpp_build
                    .file(&cpp_source_path)
                    .file(&driver_api_cpp_path)
                    .include(format!("{}/src", manifest_dir))
                    .include(rdma_include)
                    .flag("-fPIC")
                    .cpp(true)
                    .flag("-std=c++14");

                // Add CUDA include paths
                cpp_build.include(&cuda_include_path);

                // Add Python include path if available
                if let Some(include_dir) = &python_config.include_dir {
                    cpp_build.include(include_dir);
                }

                cpp_build.compile("rdmaxcel_cpp");

                // Statically link libstdc++ to avoid runtime dependency on system libstdc++
                build_utils::link_libstdcpp_static();

                // Compile EFA support with libfabric built from source
                let efa_cpp_path = format!("{}/src/rdmaxcel_efa.cpp", manifest_dir);
                if Path::new(&efa_cpp_path).exists() {
                    println!("cargo:rerun-if-env-changed=MONARCH_LIBFABRIC_TAG");
                    println!("cargo:rerun-if-changed={}", efa_cpp_path);
                    println!(
                        "cargo:rerun-if-changed={}/src/rdmaxcel_efa.h",
                        manifest_dir
                    );

                    // Build or locate libfabric
                    let libfabric_result = build_libfabric();

                    if let Some((libfabric_include, libfabric_lib_dir)) = libfabric_result {
                        // Compile rdmaxcel_efa.cpp with HAVE_LIBFABRIC
                        let mut efa_build = cc::Build::new();
                        efa_build
                            .file(&efa_cpp_path)
                            .include(format!("{}/src", manifest_dir))
                            .include(&libfabric_include)
                            .define("HAVE_LIBFABRIC", "1")
                            .flag("-fPIC")
                            .cpp(true)
                            .flag("-std=c++14");

                        efa_build.compile("rdmaxcel_efa");

                        // Export libfabric path via metadata so the final
                        // cdylib crate (monarch_extension) can link it.
                        // cargo:rustc-link-arg from a lib crate does NOT
                        // propagate to the cdylib link step.
                        let libfabric_a = format!("{}/libfabric.a", libfabric_lib_dir);
                        println!("cargo:metadata=LIBFABRIC_A={}", libfabric_a);

                        // Link libfabric's private dependencies
                        // (ibverbs + efa are already linked by monarch_cpp_static_libs)
                        // (rt, pthread, dl are already linked earlier for cudart_static)
                        println!("cargo:rustc-link-lib=numa");
                        println!("cargo:rustc-link-lib=uuid");
                        println!("cargo:rustc-link-lib=hwloc");
                        println!("cargo:rustc-link-lib=rdmacm");
                        println!("cargo:rustc-link-lib=nl-3");
                        println!("cargo:rustc-link-lib=nl-route-3");
                        println!("cargo:rustc-link-lib=atomic");

                        println!("cargo:rustc-cfg=feature=\"efa\"");
                        println!("cargo:rustc-check-cfg=cfg(feature, values(\"efa\"))");

                        println!(
                            "cargo:warning=Statically linked libfabric from {}",
                            libfabric_lib_dir
                        );
                    } else {
                        // libfabric not available — compile stubs
                        let mut efa_build = cc::Build::new();
                        efa_build
                            .file(&efa_cpp_path)
                            .include(format!("{}/src", manifest_dir))
                            .flag("-fPIC")
                            .cpp(true)
                            .flag("-std=c++14");

                        efa_build.compile("rdmaxcel_efa");
                    }
                }
            } else {
                if !Path::new(&cpp_source_path).exists() {
                    panic!("C++ source file not found at {}", cpp_source_path);
                }
                if !Path::new(&driver_api_cpp_path).exists() {
                    panic!(
                        "Driver API C++ source file not found at {}",
                        driver_api_cpp_path
                    );
                }
            }
            // Compile the CUDA source file
            let cuda_source_path = format!("{}/src/rdmaxcel.cu", manifest_dir);
            if Path::new(&cuda_source_path).exists() {
                // Use the CUDA home path we already validated
                let nvcc_path = format!("{}/bin/nvcc", cuda_home);

                // Set up fixed output directory - use a predictable path instead of dynamic OUT_DIR
                let cuda_build_dir = format!("{}/target/cuda_build", manifest_dir);
                std::fs::create_dir_all(&cuda_build_dir)
                    .expect("Failed to create CUDA build directory");

                let cuda_obj_path = format!("{}/rdmaxcel_cuda.o", cuda_build_dir);
                let cuda_lib_path = format!("{}/librdmaxcel_cuda.a", cuda_build_dir);

                // Use nvcc to compile the CUDA file
                let nvcc_output = std::process::Command::new(&nvcc_path)
                    .args([
                        "-c",
                        &cuda_source_path,
                        "-o",
                        &cuda_obj_path,
                        "--compiler-options",
                        "-fPIC",
                        "-std=c++14",
                        "--expt-extended-lambda",
                        "-Xcompiler",
                        "-fPIC",
                        &format!("-I{}", cuda_include_path),
                        &format!("-I{}/src", manifest_dir),
                        &format!("-I{}", rdma_include),
                    ])
                    .output();

                match nvcc_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
                            eprintln!("nvcc stdout: {}", String::from_utf8_lossy(&output.stdout));
                            panic!("Failed to compile CUDA source with nvcc");
                        }
                        println!("cargo:rerun-if-changed={}", cuda_source_path);
                    }
                    Err(e) => {
                        eprintln!("Failed to run nvcc: {}", e);
                        panic!("nvcc not found or failed to execute");
                    }
                }

                // Create static library from the compiled CUDA object
                let ar_output = std::process::Command::new("ar")
                    .args(["rcs", &cuda_lib_path, &cuda_obj_path])
                    .output();

                match ar_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("ar stderr: {}", String::from_utf8_lossy(&output.stderr));
                            panic!("Failed to create CUDA static library with ar");
                        }
                        // Emit metadata so dependent crates can find this library
                        println!("cargo:rustc-link-lib=static=rdmaxcel_cuda");
                        println!("cargo:rustc-link-search=native={}", cuda_build_dir);

                        // Copy the library to OUT_DIR as well for Cargo dependency mechanism
                        if let Err(e) =
                            std::fs::copy(&cuda_lib_path, format!("{}/librdmaxcel_cuda.a", out_dir))
                        {
                            eprintln!("Warning: Failed to copy CUDA library to OUT_DIR: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to run ar: {}", e);
                        panic!("ar not found or failed to execute");
                    }
                }
            } else {
                panic!("CUDA source file not found at {}", cuda_source_path);
            }
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
