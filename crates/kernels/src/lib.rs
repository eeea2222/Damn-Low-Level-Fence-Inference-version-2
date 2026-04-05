pub const QK_K: usize = 256;
pub const QK8_0: usize = 32;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q6KMapping {
    pub ql_idx: usize,
    pub ql_shift: u8,
    pub qh_idx: usize,
    pub qh_shift: u8,
}

pub fn q6k_swizzled_mapping(i: usize) -> Q6KMapping {
    let half = i / 128;
    let rem = i % 128;
    let col = rem / 32;
    let l = rem % 32;

    let mut ql_idx = 64 * half;
    let qh_idx = 32 * half + l;
    let mut ql_shift = 0;
    let qh_shift;

    if col == 0 {
        ql_idx += l;
        qh_shift = 0;
    } else if col == 1 {
        ql_idx += l + 32;
        qh_shift = 2;
    } else if col == 2 {
        ql_idx += l;
        ql_shift = 4;
        qh_shift = 4;
    } else {
        ql_idx += l + 32;
        ql_shift = 4;
        qh_shift = 6;
    }

    Q6KMapping {
        ql_idx,
        ql_shift,
        qh_idx,
        qh_shift,
    }
}

pub fn q6k_extract_signed(block: &BlockQ6K, i: usize) -> i8 {
    let m = q6k_swizzled_mapping(i);
    let lo = (block.ql[m.ql_idx] >> m.ql_shift) & 0xF;
    let hi = (block.qh[m.qh_idx] >> m.qh_shift) & 0x3;
    ((lo | (hi << 4)) as i16 - 32) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6k_layout_size_matches() {
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
    }

    #[test]
    fn q8_0_layout_size_matches() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn q6k_mapping_stays_in_bounds() {
        for i in 0..256 {
            let m = q6k_swizzled_mapping(i);
            assert!(m.ql_idx < 128);
            assert!(m.qh_idx < 64);
            assert!(m.ql_shift == 0 || m.ql_shift == 4);
            assert!(m.qh_shift <= 6 && m.qh_shift % 2 == 0);
        }
    }

    #[test]
    fn q6k_extract_matches_reference_formula() {
        let mut b = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: 0,
        };
        for (i, v) in b.ql.iter_mut().enumerate() {
            *v = ((i * 3 + 7) & 0xFF) as u8;
        }
        for (i, v) in b.qh.iter_mut().enumerate() {
            *v = ((i * 5 + 11) & 0xFF) as u8;
        }

        for i in 0..256 {
            let m = q6k_swizzled_mapping(i);
            let lo = (b.ql[m.ql_idx] >> m.ql_shift) & 0xF;
            let hi = (b.qh[m.qh_idx] >> m.qh_shift) & 0x3;
            let reference = (lo | (hi << 4)) as i16 - 32;
            assert_eq!(q6k_extract_signed(&b, i) as i16, reference);
        }
    }
}
