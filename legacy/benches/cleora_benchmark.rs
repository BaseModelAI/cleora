use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fnv::FnvHasher;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use twox_hash::XxHash64;

fn default_hash(entity: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(entity.as_bytes());
    hasher.finish()
}

fn xx_hash(entity: &str) -> u64 {
    let mut hasher = XxHash64::default();
    hasher.write(entity.as_bytes());
    hasher.finish()
}

fn fnv_hash(entity: &str) -> u64 {
    let mut hasher = FnvHasher::default();
    hasher.write(entity.as_bytes());
    hasher.finish()
}

fn hash_benchmark(c: &mut Criterion) {
    c.bench_function("hash", |b| b.iter(|| fnv_hash(black_box("cleora"))));
}

fn bench_hashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hashing");
    for s in ["Poland", "Germany", "USA", "United Kingdom", "Norway"].iter() {
        group.bench_with_input(BenchmarkId::new("Default", s), s, |b, s| {
            b.iter(|| default_hash(s))
        });
        group.bench_with_input(BenchmarkId::new("XXHash", s), s, |b, s| {
            b.iter(|| xx_hash(s))
        });
        group.bench_with_input(BenchmarkId::new("FnvHash", s), s, |b, s| {
            b.iter(|| fnv_hash(s))
        });
    }
    group.finish();
}

struct CartesianProduct {
    lengths: Vec<u32>,
    indices: Vec<u32>,
}

impl CartesianProduct {
    fn new(lengths: Vec<u32>) -> CartesianProduct {
        let indices = vec![0; lengths.len()];
        CartesianProduct { lengths, indices }
    }
}

impl Iterator for CartesianProduct {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.indices.clone();
        let len = self.indices.len();
        for i in (0..len).rev() {
            if self.indices[i] == (self.lengths[i] - 1) {
                self.indices[i] = 0;
                if i == 0 {
                    return None;
                }
            } else {
                self.indices[i] += 1;
                break;
            }
        }
        Some(result)
    }
}

fn generate_combinations_with_length(
    hashes: Vec<Vec<u64>>,
    lens: Vec<u32>,
    transient_lens: Vec<u32>,
) -> Vec<Vec<u64>> {
    let row_length = lens.len();
    let mut combinations = 1;
    for &len in &lens {
        combinations *= len;
    }

    let mut transient_combinations = 1;
    for transient_len in transient_lens {
        transient_combinations *= transient_len;
    }

    let total_combinations = u64::from(combinations * transient_combinations);

    let mut result: Vec<Vec<u64>> = Vec::with_capacity(combinations as usize);
    let cartesian = CartesianProduct::new(lens);
    let mut counter = 0;

    for indices in cartesian {
        let mut arr: Vec<u64> = Vec::with_capacity(row_length + 1);
        arr.push(total_combinations);
        let hashes_length = hashes.len();
        for i in 0..hashes_length {
            let id = indices[i];
            let value = hashes.get(i).unwrap().get(id as usize).unwrap();
            arr.push(*value);
        }
        result.insert(counter, arr);
        counter += 1;
    }

    result
}

fn generate_combinations_with_length_benchmark(c: &mut Criterion) {
    let hashes = vec![
        vec![
            12528106613309397869,
            9708327007652588651,
            14980293948133487802,
            12266831465718424827,
            17286486014462130850,
            11758309849656381133,
            10347099512938872293,
            804562942093240192,
            3059164883323983321,
        ],
        vec![
            12528106613309397869,
            9708327007652588651,
            14980293948133487802,
            12266831465718424827,
            17286486014462130850,
            11758309849656381133,
            10347099512938872293,
            804562942093240192,
            3059164883323983321,
        ],
    ];
    let lens = vec![9, 9];
    let transient_lens = vec![1];
    c.bench_function("generate_combinations_with_length", |b| {
        b.iter(|| {
            generate_combinations_with_length(
                black_box(hashes.clone()),
                black_box(lens.clone()),
                black_box(transient_lens.clone()),
            )
        })
    });
}

criterion_group!(
    benches,
    generate_combinations_with_length_benchmark,
    bench_hashes
);
criterion_main!(benches);
