# 1.2.0 (March 17, 2022)

### Added
- Use default hasher for vector init. ([#47]).

[#47]: https://github.com/Synerise/cleora/pull/47


# 1.1.1 (May 14, 2021)

### Added
- Init embedding with seed during training ([#27]).

[#27]: https://github.com/Synerise/cleora/pull/27


# 1.1.0 (December 23, 2020)

### Changed
- Bumped `env_logger` to `0.8.2`, `smallvec` to `1.5.1`, removed `fnv` hasher ([#11]).

[#11]: https://github.com/Synerise/cleora/pull/11

### Added
- Tests (snapshots) for in-memory and memory-mapped files calculations of embeddings ([#12]).
- Support for `NumPy` output format (available via `--output-format` program argument) ([#15]).
- Jupyter notebooks with experiments ([#16]).

[#12]: https://github.com/Synerise/cleora/pull/12
[#15]: https://github.com/Synerise/cleora/pull/15
[#16]: https://github.com/Synerise/cleora/pull/16

### Improved
- Used `vector` for `hash_to_id` mappings, non-allocating cartesian product, `ryu` crate for faster write ([#13]).
- Sparse Matrix refactor (cleanup, simplification, using iter, speedup). Use Cargo.toml data for clap crate ([#17]).
- Unify and simplify embeddings calculation for in-memory and mmap matrices ([#18]).

[#13]: https://github.com/Synerise/cleora/pull/13
[#17]: https://github.com/Synerise/cleora/pull/17
[#18]: https://github.com/Synerise/cleora/pull/18


# 1.0.1 (November 23, 2020)

### Fixed
- Skip reading invalid UTF-8 line ([#8]).
- Fix clippy warnings ([#7]).

[#8]: https://github.com/Synerise/cleora/pull/8
[#7]: https://github.com/Synerise/cleora/pull/7

### Added
- JSON support ([#3]).
- Snapshot testing ([#5]).

[#3]: https://github.com/Synerise/cleora/pull/3
[#5]: https://github.com/Synerise/cleora/pull/5


# 1.0.0 (November 6, 2020)

- Initial release.