use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let source_dir = Path::new("stylegan2-ada-pytorch/torch_utils/ops");

    let mut cargo_commands = vec![];
    torch_build::build_cpp(
        true,
        true,
        Some(&mut cargo_commands),
        [source_dir.join("upfirdn2d.cpp")],
    )?
    .try_compile("upfirdn2d_cpp")?;
    torch_build::build_cuda(
        true,
        Some(&mut cargo_commands),
        [source_dir.join("upfirdn2d.cu")],
    )?
    .try_compile("upfirdn2d_cuda")?;

    cargo_commands.iter().for_each(|command| {
        println!("{}", command);
    });

    Ok(())
}
