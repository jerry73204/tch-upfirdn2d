use tch::{nn, Device};
use tch_upfirdn2d::upfirdn2d;

#[test]
fn upfirdn2d_test() {
    let vs = nn::VarStore::new(Device::Cuda(0));
    let root = vs.root();

    let x = root.randn_standard("x", &[5, 3, 16, 16]);
    let f = root.randn_standard("f", &[3, 3]);
    let out = upfirdn2d(&x, &f, 1, 1, 1, 1, 0, 0, 0, 0, true, 1.0);
    assert_eq!(out.size(), [5, 3, 14, 14]);
}
