use num::complex::Complex;
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

trait Field:
    Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> + Div<Output = Self> + Sized
{
    fn one() -> Self;
}
impl Field for f64 {
    fn one() -> Self {
        1.0
    }
}
impl Field for Complex<f64> {
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}

fn main() {}

fn combine_dfts_time<T: Field + Copy>(y_1: &mut [T], y_2: &mut [T], w_n: &[T], n: usize) {
    let len = y_1.len();
    let factor = n / len / 2;
    y_1.iter_mut()
        .zip(y_2.iter_mut())
        .enumerate()
        .for_each(|(i, (y_1, y_2))| {
            let w = w_n[i * factor];
            let output_i = *y_1 + *y_2 * w;
            let output_i_len = *y_1 - *y_2 * w;
            *y_1 = output_i;
            *y_2 = output_i_len;
        });
}

fn decombine_dfts_time<T: Field + Copy>(y_1: &mut [T], y_2: &mut [T], w_n: &[T], n: usize) {
    let len = y_1.len();
    let factor = n / len / 2;
    y_1.iter_mut()
        .zip(y_2.iter_mut())
        .enumerate()
        .for_each(|(i, (y_1, y_2))| {
            let w = w_n[i * factor];
            let two = T::one() + T::one();
            let output_i = (*y_1 + *y_2) / two;
            let output_i_len = (*y_1 - *y_2) / two / w;
            *y_1 = output_i;
            *y_2 = output_i_len;
        });
}
fn combine_dfts_freq<T: Field + Copy>(y_1: &mut [T], y_2: &mut [T], w_n: &[T], n: usize) {
    let len = y_1.len();
    let factor = n / len / 2;
    y_1.iter_mut()
        .zip(y_2.iter_mut())
        .enumerate()
        .for_each(|(i, (y_1, y_2))| {
            let w = w_n[i * factor];
            let output_i = *y_1 + *y_2;
            let output_i_len = (*y_1 - *y_2) * w;
            *y_1 = output_i;
            *y_2 = output_i_len;
        });
}

fn decombine_dfts_freq<T: Field + Copy>(y_1: &mut [T], y_2: &mut [T], w_n: &[T], n: usize) {
    let len = y_1.len();
    let factor = n / len / 2;
    y_1.iter_mut()
        .zip(y_2.iter_mut())
        .enumerate()
        .for_each(|(i, (y_1, y_2))| {
            let w = w_n[i * factor];
            let two = T::one() + T::one();
            let output_i = (*y_1 + *y_2 / w) / two;
            let output_i_len = (*y_1 - *y_2 / w) / two;
            *y_1 = output_i;
            *y_2 = output_i_len;
        });
}

fn bit_reverse_copy<T, I: ?Sized>(input: &I) -> Vec<T>
where
    T: Copy + Default,
    for<'a> &'a I: IntoIterator<Item = &'a T>,
    for<'a> <&'a I as std::iter::IntoIterator>::IntoIter: ExactSizeIterator,
{
    let n = input.into_iter().len();
    let mut output = vec![T::default(); n];
    let n_bits = (n as f64).log2() as usize;
    for (i, val) in input.into_iter().enumerate() {
        let reversed = i.reverse_bits() >> (std::mem::size_of::<usize>() * 8 - n_bits);
        output[reversed] = *val;
    }
    output
}
fn bit_reverse_inplace<T: Field + Copy + Default>(input: &mut [T]) {
    let n = input.len();
    let n_bits = (n as f64).log2() as usize;
    for i in 0..n {
        let reversed = i.reverse_bits() >> (std::mem::size_of::<usize>() * 8 - n_bits);
        if i < reversed {
            input.swap(i, reversed);
        }
    }
}
trait FFT<T>
where
    T: Field + Copy + Default,
    for<'a> &'a Self: IntoIterator<Item = &'a T>,
    // Self: std::convert::AsRef<[T]>,
    for<'a> <&'a Self as std::iter::IntoIterator>::IntoIter: ExactSizeIterator,
    Self: Sized,
{
    fn dft_decimation_in_time(&self) -> Vec<T> {
        assert!(is_power_of_two(self.into_iter().len()));
        let n = self.into_iter().len();
        let mut output = bit_reverse_copy(self);

        // https://www.alwayslearn.com/DFT%20and%20FFT%20Tutorial/DFTandFFT_FFT_Butterfly_8_Input.html
        // https://www.rcet.org.in/uploads/academics/rohini_39437371635.pdf
        let w_n = Self::compute_half_w_vec(n);
        for stage in 0..(n as f64).log2() as usize {
            output
                .chunks_mut(2_usize.pow((stage + 1) as u32))
                .for_each(|chunk| {
                    let (y_1, y_2) = chunk.split_at_mut(chunk.len() / 2);
                    combine_dfts_time(y_1, y_2, &w_n, n);
                });
        }
        output
    }
    fn compute_half_w_vec(n: usize) -> Vec<T>;
    fn inverse_dft_decimation_in_time(&self) -> Vec<T> {
        // check that input_sec is a power of 2
        assert!(is_power_of_two(self.into_iter().len()));
        let n = self.into_iter().len();
        let mut output = self.into_iter().cloned().collect::<Vec<T>>();
        let w_n = Self::compute_half_w_vec(n);
        // https://www.rcet.org.in/uploads/academics/rohini_87453928097.pdf
        for stage in (0..(n as f64).log2() as usize).rev() {
            output
                .chunks_mut(2_usize.pow((stage + 1) as u32))
                .for_each(|chunk| {
                    let (y_1, y_2) = chunk.split_at_mut(chunk.len() / 2);
                    decombine_dfts_time(y_1, y_2, &w_n, n);
                });
        }
        bit_reverse_inplace(&mut output);
        output
    }
    fn dft_decimation_in_freq(&self) -> Vec<T> {
        assert!(is_power_of_two(self.into_iter().len()));
        let n = self.into_iter().len();
        let mut output = self.into_iter().cloned().collect::<Vec<T>>();
        let w_n = Self::compute_half_w_vec(n);
        // https://www.rcet.org.in/uploads/academics/rohini_87453928097.pdf
        for stage in (0..(n as f64).log2() as usize).rev() {
            output
                .chunks_mut(2_usize.pow((stage + 1) as u32))
                .for_each(|chunk| {
                    let (y_1, y_2) = chunk.split_at_mut(chunk.len() / 2);
                    combine_dfts_freq(y_1, y_2, &w_n, n);
                });
        }
        bit_reverse_inplace(&mut output);
        output
    }

    fn inverse_dft_decimation_in_freq(&self) -> Vec<T> {
        assert!(is_power_of_two(self.into_iter().len()));
        let n = self.into_iter().len();
        let mut output = bit_reverse_copy(self);
        let w_n = Self::compute_half_w_vec(n);
        // https://www.rcet.org.in/uploads/academics/rohini_87453928097.pdf
        for stage in 0..(n as f64).log2() as usize {
            output
                .chunks_mut(2_usize.pow((stage + 1) as u32))
                .for_each(|chunk| {
                    let (y_1, y_2) = chunk.split_at_mut(chunk.len() / 2);
                    decombine_dfts_freq(y_1, y_2, &w_n, n);
                });
        }
        output
    }
}
impl FFT<Complex<f64>> for Vec<Complex<f64>> {
    fn compute_half_w_vec(n: usize) -> Vec<Complex<f64>> {
        let w = Complex::new(0.0, -2.0 * PI / n as f64).exp();
        let mut res: Vec<Complex<f64>> = Vec::with_capacity(n / 2);
        let mut prod = Complex::new(1.0, 0.0);
        (0..n / 2).for_each(|_| {
            res.push(prod);
            prod *= w;
        });
        res
    }
}

fn is_power_of_two(len: usize) -> bool {
    len.count_ones() == 1
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use rand::Rng;

    use super::*;

    trait ImpreciseEq {
        fn imprecise_eq(&self, other: &Self, eps: f64) -> bool;
    }
    impl<T> ImpreciseEq for Complex<T>
    where
        T: ImpreciseEq,
        T: Copy,
    {
        fn imprecise_eq(&self, other: &Self, eps: f64) -> bool {
            self.re.imprecise_eq(&other.re, eps) && self.im.imprecise_eq(&other.im, eps)
        }
    }
    impl ImpreciseEq for f64 {
        fn imprecise_eq(&self, other: &Self, eps: f64) -> bool {
            (self - other).abs() < eps
        }
    }
    impl<T> ImpreciseEq for Vec<T>
    where
        T: ImpreciseEq,
        T: Copy,
    {
        fn imprecise_eq(&self, other: &Self, eps: f64) -> bool {
            self.iter()
                .zip(other.iter())
                .all(|(&x, &y)| x.imprecise_eq(&y, eps))
        }
    }
    fn to_complex<T, V>(input: &[T]) -> Vec<Complex<V>>
    where
        T: Copy,
        V: std::convert::From<T>,
        V: std::convert::From<i32>,
    {
        input
            .iter()
            .map(|&x| Complex::new(x.into(), 0.into()))
            .collect()
    }
    fn dft_matrix(n: usize) -> Array2<Complex<f64>> {
        let w = Complex::new(0.0, -2.0 * PI / n as f64).exp();
        let mut dft_matrix = Array2::from_elem((n, n), Complex::new(0.0, 0.0));

        for i in 0..n {
            for j in 0..n {
                dft_matrix[(i, j)] = w.powf((i * j) as f64);
            }
        }
        dft_matrix
    }
    pub fn dft_simple(input_sec: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let n = input_sec.len();
        let dft_matr = dft_matrix(n);
        // multiply input with dft matrix
        let input_sec = Array1::from(input_sec.to_vec());
        dft_matr.dot(&input_sec).to_vec()
    }

    fn two_point_dft(x: &mut [Complex<f64>]) {
        let x1 = x[0] + x[1];
        let x2 = x[0] - x[1];
        x[0..2].copy_from_slice(&[x1, x2]);
    }
    fn assert_almost_eq<T>(left: &T, right: &T, eps: f64)
    where
        T: ImpreciseEq,
        T: std::fmt::Debug,
    {
        assert!(
            left.imprecise_eq(right, eps),
            "left: {:?}, right: {:?}",
            left,
            right
        )
    }
    #[test]
    fn test_simple_dft_manual() {
        let input = to_complex(&[19, 20, 13, 11, 9, 42, 14, 6]);
        let j = Complex::new(0.0, 1.0);
        let output: Vec<Complex<f64>> = vec![
            134.0 + 0.0 * j,
            -9.092 + 13.021 * j,
            1.0 - 45.0 * j,
            29.092 + 11.021 * j,
            -24.0 + 0.0 * j,
            29.092 - 11.021 * j,
            1.0 + 45.0 * j,
            -9.092 - 13.021 * j,
        ];
        let res = dft_simple(&input);
        assert_almost_eq(&res, &output, 1e-3);
    }
    #[test]
    fn test_dft_decimation_in_time() {
        let input = random_vec(16);
        let res = input.dft_decimation_in_time();
        assert_almost_eq(&res, &dft_simple(&input), 1e-6);
    }

    fn random_vec(len: usize) -> Vec<Complex<f64>> {
        let mut rng = rand::thread_rng();
        let input: Vec<Complex<f64>> = (0..len)
            .map(|_| Complex::new(rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
            .collect();
        input
    }
    #[test]
    fn test_dft_decimation_in_freq() {
        let input = random_vec(16);
        let res = input.dft_decimation_in_freq();
        assert_almost_eq(&res, &dft_simple(&input), 1e-6);
    }
    #[test]
    fn test_two_point_dft() {
        let x_1 = Complex::new(19.0, 0.0);
        let x_2 = Complex::new(20.0, 0.0);
        let mut output = vec![x_1, x_2];
        two_point_dft(&mut output);
        assert_almost_eq(&output, &dft_simple(&[x_1, x_2]), 1e-6);
    }
    #[test]
    fn test_combine_dft_for_two_points() {
        let x_1 = Complex::new(19.0, 0.0);
        let x_2 = Complex::new(20.0, 0.0);
        let mut output = vec![x_1, x_2];
        let (y1, y2) = output.split_at_mut(1);
        let w_n = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        combine_dfts_time(y1, y2, &w_n, 2);
        assert_almost_eq(&output, &dft_simple(&[x_1, x_2]), 1e-6);
    }
    #[test]
    fn test_bit_reverse_copy() {
        let input = to_complex(&[19, 20, 13, 11, 9, 42, 14, 6]);
        let output = bit_reverse_copy::<Complex<f64>, [Complex<f64>]>(&input);
        let expected = to_complex(&[19, 9, 13, 14, 20, 42, 11, 6]);
        assert_eq!(output, expected);
    }
    #[test]
    fn test_bit_reverse_inplace() {
        let mut input = to_complex(&[19, 20, 13, 11, 9, 42, 14, 6]);
        bit_reverse_inplace(&mut input);
        let expected = to_complex(&[19, 9, 13, 14, 20, 42, 11, 6]);
        assert_eq!(input, expected);
    }
    #[test]
    fn test_inverse_dft_decimation_in_time() {
        let input = to_complex(&[19, 20, 13, 11, 9, 42, 14, 6]);
        let res = input
            .dft_decimation_in_time()
            .inverse_dft_decimation_in_time();
        assert_almost_eq(&res, &input, 1e-6);
    }
    #[test]
    fn test_inverse_dft_decimation_in_freq() {
        let input = to_complex(&[19, 20, 13, 11, 9, 42, 14, 6]);
        let res = input
            .dft_decimation_in_freq()
            .inverse_dft_decimation_in_freq();
        assert_almost_eq(&res, &input, 1e-6);
    }
    #[test]
    fn almost_eq() {
        let a = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let b = vec![Complex::new(1.0, 0.0), Complex::new(2.01, 0.0)];
        assert_almost_eq(&a, &b, 1e-1);
    }
    #[test]
    #[should_panic]
    fn almost_eq_panic() {
        let a = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let b = vec![Complex::new(1.0, 0.0), Complex::new(2.1, 0.0)];
        assert_almost_eq(&a, &b, 1e-10);
    }
}
