use anyhow::{anyhow, Result};

const CSRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc");
const BINDINGS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings.rs");
const SOURCE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/stylegan2-ada-pytorch/torch_utils/ops"
);

fn main() -> Result<()> {
    #[cfg(feature = "codegen")]
    codegen()?;

    #[cfg(feature = "link")]
    link()?;

    // Set re-run conditions
    println!("cargo:rerun-if-changed={}/wrapper.hpp", CSRC_DIR);
    println!("cargo:rerun-if-changed={}/wrapper.cpp", CSRC_DIR);

    Ok(())
}

#[cfg(feature = "codegen")]
fn codegen() -> Result<()> {
    let library = torch_build::probe_libtorch()?;

    let mut builder = bindgen::Builder::default();

    // add libtorch related includes
    for path in library.include_paths(true)? {
        builder = builder.clang_args(["-Xcompiler", &format!("-I{}", path.display())]);
    }

    // add wrapper code
    builder = builder
        .clang_args(["-Xcompiler", &format!("-I{}", SOURCE_DIR)])
        .header(format!("{}/wrapper.hpp", CSRC_DIR));

    // set allowlist
    builder = builder.allowlist_function("upfirdn2d_ffi");

    let bindings = builder
        .generate()
        .map_err(|_| anyhow!("unable to generate bindings"))?;

    bindings.write_to_file(BINDINGS_PATH)?;

    Ok(())
}

fn link() -> Result<()> {
    let mut cargo_commands = vec![];

    {
        let mut build = cc::Build::new();
        build.include(SOURCE_DIR);
        torch_build::build_cuda(
            &mut build,
            true,
            Some(&mut cargo_commands),
            [format!("{}/wrapper.cpp", CSRC_DIR)],
        )?;
        build.try_compile("upfirdn2d_cpp")?;
    }

    {
        let mut build = cc::Build::new();
        build.include(SOURCE_DIR);
        torch_build::build_cuda(
            &mut build,
            true,
            Some(&mut cargo_commands),
            [format!("{}/wrapper.cu", CSRC_DIR)],
        )?;
        build.try_compile("upfirdn2d_cuda")?;
    }

    cargo_commands.iter().for_each(|command| {
        println!("{}", command);
    });

    Ok(())
}
