use common::FenceError;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Bf16 = 30,
    Unknown = u32::MAX,
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::Bf16,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
    Unknown = u32::MAX,
}

impl From<u32> for GgufValueType {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GgmlTypeInfo {
    pub block_size: u32,
    pub type_size: u32,
}

pub fn ggml_type_info(ty: GgmlType) -> GgmlTypeInfo {
    match ty {
        GgmlType::F32 => GgmlTypeInfo {
            block_size: 1,
            type_size: 4,
        },
        GgmlType::F16 => GgmlTypeInfo {
            block_size: 1,
            type_size: 2,
        },
        GgmlType::Q4_0 => GgmlTypeInfo {
            block_size: 32,
            type_size: 18,
        },
        GgmlType::Q4_1 => GgmlTypeInfo {
            block_size: 32,
            type_size: 20,
        },
        GgmlType::Q5_0 => GgmlTypeInfo {
            block_size: 32,
            type_size: 22,
        },
        GgmlType::Q5_1 => GgmlTypeInfo {
            block_size: 32,
            type_size: 24,
        },
        GgmlType::Q8_0 => GgmlTypeInfo {
            block_size: 32,
            type_size: 34,
        },
        GgmlType::Q8_1 => GgmlTypeInfo {
            block_size: 32,
            type_size: 40,
        },
        GgmlType::Q2_K => GgmlTypeInfo {
            block_size: 256,
            type_size: 84,
        },
        GgmlType::Q3_K => GgmlTypeInfo {
            block_size: 256,
            type_size: 110,
        },
        GgmlType::Q4_K => GgmlTypeInfo {
            block_size: 256,
            type_size: 144,
        },
        GgmlType::Q5_K => GgmlTypeInfo {
            block_size: 256,
            type_size: 176,
        },
        GgmlType::Q6_K => GgmlTypeInfo {
            block_size: 256,
            type_size: 210,
        },
        GgmlType::Q8_K => GgmlTypeInfo {
            block_size: 256,
            type_size: 292,
        },
        GgmlType::Bf16 => GgmlTypeInfo {
            block_size: 1,
            type_size: 2,
        },
        GgmlType::I8 => GgmlTypeInfo {
            block_size: 1,
            type_size: 1,
        },
        GgmlType::I16 => GgmlTypeInfo {
            block_size: 1,
            type_size: 2,
        },
        GgmlType::I32 => GgmlTypeInfo {
            block_size: 1,
            type_size: 4,
        },
        GgmlType::I64 | GgmlType::F64 => GgmlTypeInfo {
            block_size: 1,
            type_size: 8,
        },
        GgmlType::Unknown => GgmlTypeInfo {
            block_size: 1,
            type_size: 4,
        },
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct MetadataKv {
    pub key: String,
    pub ty: GgufValueType,
    pub value: MetadataValue,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub ty: GgmlType,
    pub offset: u64,
    pub data_size_bytes: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims[..self.n_dims as usize].iter().product()
    }
}

#[derive(Debug)]
pub struct GgufFile {
    mmap: Mmap,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: u32,
    tensor_data_offset: usize,
    metadata: Vec<MetadataKv>,
    tensors: Vec<TensorInfo>,
    metadata_index: HashMap<String, usize>,
    tensor_index: HashMap<String, usize>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FenceError> {
        let file = File::open(path)?;
        // SAFETY: File is opened read-only and mapped immutably for process lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut p = Parser::new(&mmap);

        let magic = p.read_u32()?;
        if magic != 0x4655_4747 {
            return Err(FenceError::Parse("invalid GGUF magic".into()));
        }
        let version = p.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(FenceError::Unsupported(format!(
                "unsupported GGUF version {version}"
            )));
        }

        let tensor_count = p.read_u64()?;
        let metadata_kv_count = p.read_u64()?;

        let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
        let mut metadata_index = HashMap::new();
        let mut alignment = 32_u32;

        for _ in 0..metadata_kv_count {
            let key = p.read_string()?;
            let ty: GgufValueType = p.read_u32()?.into();
            if let Some(value) = p.read_value(ty)? {
                if key == "general.alignment"
                    && let MetadataValue::Uint32(a) = value
                    && a != 0
                {
                    alignment = a;
                }
                metadata_index.insert(key.clone(), metadata.len());
                metadata.push(MetadataKv { key, ty, value });
            }
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut tensor_index = HashMap::new();
        for _ in 0..tensor_count {
            let name = p.read_string()?;
            let n_dims = p.read_u32()?;
            if n_dims > 4 {
                return Err(FenceError::Parse(format!(
                    "tensor {name} has too many dimensions: {n_dims}"
                )));
            }
            let mut dims = [1_u64; 4];
            for d in dims.iter_mut().take(n_dims as usize) {
                *d = p.read_u64()?;
            }
            let ty: GgmlType = p.read_u32()?.into();
            let offset = p.read_u64()?;

            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                ty,
                offset,
                data_size_bytes: 0,
            });
        }

        let cursor = p.cursor();
        let tensor_data_offset =
            cursor + ((alignment as usize - (cursor % alignment as usize)) % alignment as usize);

        for t in &mut tensors {
            let abs_offset = tensor_data_offset + t.offset as usize;
            if abs_offset >= mmap.len() {
                return Err(FenceError::Parse(format!(
                    "tensor '{}' offset out of bounds",
                    t.name
                )));
            }
            let info = ggml_type_info(t.ty);
            let n_elem = t.n_elements();
            let n_blocks = n_elem.div_ceil(info.block_size as u64);
            t.data_size_bytes = (n_blocks * info.type_size as u64) as usize;
        }

        Ok(Self {
            mmap,
            version,
            tensor_count,
            metadata_kv_count,
            alignment,
            tensor_data_offset,
            metadata,
            tensors,
            metadata_index,
            tensor_index,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }
    pub fn metadata_count(&self) -> u64 {
        self.metadata_kv_count
    }
    pub fn metadata(&self) -> &[MetadataKv] {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn data_offset(&self) -> usize {
        self.tensor_data_offset
    }
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index
            .get(name)
            .and_then(|&i| self.tensors.get(i))
    }

    pub fn find_metadata(&self, key: &str) -> Option<&MetadataKv> {
        self.metadata_index
            .get(key)
            .and_then(|&i| self.metadata.get(i))
    }

    pub fn tensor_bytes(&self, tensor: &TensorInfo) -> Option<&[u8]> {
        let start = self
            .tensor_data_offset
            .checked_add(tensor.offset as usize)?;
        let end = start.checked_add(tensor.data_size_bytes)?;
        self.mmap.get(start..end)
    }

    pub fn get_string(&self, key: &str) -> Result<String, FenceError> {
        match self
            .find_metadata(key)
            .ok_or_else(|| FenceError::MissingKey(key.to_string()))?
            .value
        {
            MetadataValue::String(ref s) => Ok(s.clone()),
            _ => Err(FenceError::TypeMismatch(format!("{key} is not string"))),
        }
    }

    pub fn get_u32(&self, key: &str) -> Result<u32, FenceError> {
        match self
            .find_metadata(key)
            .ok_or_else(|| FenceError::MissingKey(key.to_string()))?
            .value
        {
            MetadataValue::Uint32(v) => Ok(v),
            _ => Err(FenceError::TypeMismatch(format!("{key} is not u32"))),
        }
    }

    pub fn get_i32(&self, key: &str) -> Result<i32, FenceError> {
        match self
            .find_metadata(key)
            .ok_or_else(|| FenceError::MissingKey(key.to_string()))?
            .value
        {
            MetadataValue::Int32(v) => Ok(v),
            _ => Err(FenceError::TypeMismatch(format!("{key} is not i32"))),
        }
    }

    pub fn get_f32(&self, key: &str) -> Result<f32, FenceError> {
        match self
            .find_metadata(key)
            .ok_or_else(|| FenceError::MissingKey(key.to_string()))?
            .value
        {
            MetadataValue::Float32(v) => Ok(v),
            _ => Err(FenceError::TypeMismatch(format!("{key} is not f32"))),
        }
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, FenceError> {
        match self
            .find_metadata(key)
            .ok_or_else(|| FenceError::MissingKey(key.to_string()))?
            .value
        {
            MetadataValue::Bool(v) => Ok(v),
            _ => Err(FenceError::TypeMismatch(format!("{key} is not bool"))),
        }
    }

    pub fn get_u64(&self, key: &str) -> Result<u64, FenceError> {
        match self
            .find_metadata(key)
            .ok_or_else(|| FenceError::MissingKey(key.to_string()))?
            .value
        {
            MetadataValue::Uint64(v) => Ok(v),
            _ => Err(FenceError::TypeMismatch(format!("{key} is not u64"))),
        }
    }
}

struct Parser<'a> {
    data: &'a [u8],
    cursor: usize,
}

impl<'a> Parser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    fn cursor(&self) -> usize {
        self.cursor
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], FenceError> {
        let end = self
            .cursor
            .checked_add(n)
            .ok_or_else(|| FenceError::Parse("cursor overflow".into()))?;
        let out = self
            .data
            .get(self.cursor..end)
            .ok_or_else(|| FenceError::Parse("unexpected EOF".into()))?;
        self.cursor = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, FenceError> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8, FenceError> {
        Ok(self.read_u8()? as i8)
    }
    fn read_u16(&mut self) -> Result<u16, FenceError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_i16(&mut self) -> Result<i16, FenceError> {
        Ok(self.read_u16()? as i16)
    }
    fn read_u32(&mut self) -> Result<u32, FenceError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_i32(&mut self) -> Result<i32, FenceError> {
        Ok(self.read_u32()? as i32)
    }
    fn read_u64(&mut self) -> Result<u64, FenceError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_i64(&mut self) -> Result<i64, FenceError> {
        Ok(self.read_u64()? as i64)
    }
    fn read_f32(&mut self) -> Result<f32, FenceError> {
        Ok(f32::from_bits(self.read_u32()?))
    }
    fn read_f64(&mut self) -> Result<f64, FenceError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String, FenceError> {
        let len = self.read_u64()? as usize;
        let b = self.read_bytes(len)?;
        String::from_utf8(b.to_vec())
            .map_err(|e| FenceError::Parse(format!("invalid utf8 string: {e}")))
    }

    fn skip_value(&mut self, ty: GgufValueType) -> Result<(), FenceError> {
        match ty {
            GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
                self.read_bytes(1)?;
            }
            GgufValueType::Uint16 | GgufValueType::Int16 => {
                self.read_bytes(2)?;
            }
            GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
                self.read_bytes(4)?;
            }
            GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
                self.read_bytes(8)?;
            }
            GgufValueType::String => {
                let len = self.read_u64()? as usize;
                self.read_bytes(len)?;
            }
            GgufValueType::Array => {
                let arr_ty: GgufValueType = self.read_u32()?.into();
                let arr_len = self.read_u64()?;
                for _ in 0..arr_len {
                    self.skip_value(arr_ty)?;
                }
            }
            GgufValueType::Unknown => {
                return Err(FenceError::Unsupported(
                    "unknown metadata value type".into(),
                ));
            }
        }
        Ok(())
    }

    fn read_value(&mut self, ty: GgufValueType) -> Result<Option<MetadataValue>, FenceError> {
        let v = match ty {
            GgufValueType::Uint8 => MetadataValue::Uint8(self.read_u8()?),
            GgufValueType::Int8 => MetadataValue::Int8(self.read_i8()?),
            GgufValueType::Uint16 => MetadataValue::Uint16(self.read_u16()?),
            GgufValueType::Int16 => MetadataValue::Int16(self.read_i16()?),
            GgufValueType::Uint32 => MetadataValue::Uint32(self.read_u32()?),
            GgufValueType::Int32 => MetadataValue::Int32(self.read_i32()?),
            GgufValueType::Float32 => MetadataValue::Float32(self.read_f32()?),
            GgufValueType::Bool => MetadataValue::Bool(self.read_u8()? != 0),
            GgufValueType::String => MetadataValue::String(self.read_string()?),
            GgufValueType::Uint64 => MetadataValue::Uint64(self.read_u64()?),
            GgufValueType::Int64 => MetadataValue::Int64(self.read_i64()?),
            GgufValueType::Float64 => MetadataValue::Float64(self.read_f64()?),
            GgufValueType::Array => {
                let arr_ty: GgufValueType = self.read_u32()?.into();
                let arr_len = self.read_u64()?;
                for _ in 0..arr_len {
                    self.skip_value(arr_ty)?;
                }
                return Ok(None);
            }
            GgufValueType::Unknown => {
                return Err(FenceError::Unsupported(
                    "unknown metadata value type".into(),
                ));
            }
        };
        Ok(Some(v))
    }
}
