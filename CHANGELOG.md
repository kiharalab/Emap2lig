# Changelog

## [0.7.1](https://github.com/kiharalab/Emap2lig/compare/v0.7.0...v0.7.1) (2026-10-31)


### Features

* **web:** native structured job progress, single-GPU queue, and cancel
* **web:** diffusion-step progress during structure modeling
* **cli:** tqdm is supplied by the CLI; library code emits ProgressEvents


### Bug Fixes

* **web:** failed predictions update the writer failed count
* **web:** cancel during sampling is not counted as a failed batch

## [0.7.0](https://github.com/kiharalab/Emap2lig/compare/v0.6.0...v0.7.0) (2026-10-30)


### Features

* **web:** fetch prebuilt frontend from npm instead of tracking dist ([afc8820](https://github.com/kiharalab/Emap2lig/commit/afc8820903a0bdc17ae0379e0b5086510a11cb88))

## [0.6.0](https://github.com/kiharalab/Emap2lig/compare/v0.5.0...v0.6.0) (2026-10-28)


### ⚠ BREAKING CHANGES

* Emap2lig is no longer a LightningModule, LigandWriter is no longer a BasePredictionWriter, and utils.device.trainer_device_kwargs is gone.
* PairformerArgs.tri_attn_use_kernel and tri_mul_use_kernel, and the use_cuequiv arguments of the triangle layers, are removed.
* emap2lig.main, emap2lig.frag and load_config are removed. Import from the emap2lig package and build an InferenceConfig; run_structure_modeling(output_dir, ligand_records, cfg, blob_ids=None) reads blobs from output_dir and takes multiplicity from cfg. detect_ligand_objects returns the blobs directory and raises NoBlobsDetectedError / TooManyBlobsError. --ligand-list is required.

### Bug Fixes

* **web:** load the tutorial output from the repository root ([84b97c2](https://github.com/kiharalab/Emap2lig/commit/84b97c2a24f1a01ccc398a91405839cbf3c37992))


### Code Refactoring

* drop the cuequivariance dependency ([ae8724c](https://github.com/kiharalab/Emap2lig/commit/ae8724cf9ee356d986bb9ea0b9fe1fe65f34bdf4))
* drop the Lightning dependency ([0311a18](https://github.com/kiharalab/Emap2lig/commit/0311a187d4532ab410b3e737d9fe1db667aad338))
* replace Hydra config with built-in defaults ([ce3bacf](https://github.com/kiharalab/Emap2lig/commit/ce3bacf38f194842fafe0741f0d6c6d5ecf5b8d5))
* replace loguru with stdlib logging ([cb199a0](https://github.com/kiharalab/Emap2lig/commit/cb199a0c7453b7b62db1c8a4dee423d8680b5941))
* split main.py into pipeline, cli and utils packages ([bb68c89](https://github.com/kiharalab/Emap2lig/commit/bb68c89102a59e1986097046c9f30714d4bdb910))


### Documentation

* describe the package-level API, pipeline and cli layout ([4573e42](https://github.com/kiharalab/Emap2lig/commit/4573e429d182fd1f96f3592f9ca1dbd91eb8888e))
* **readme:** add v0.5.0 to latest updates ([56ff50e](https://github.com/kiharalab/Emap2lig/commit/56ff50ed9f12d7ec9e581a1cb9a07a703897833a))
* **readme:** add v0.6.0 to latest updates ([8970ad5](https://github.com/kiharalab/Emap2lig/commit/8970ad5389f73dbb7c085965e693d9ab67453231))

## [0.5.0](https://github.com/kiharalab/Emap2lig/compare/v0.4.4...v0.5.0) (2026-10-28)


### Features

* support Python 3.13 and 3.14 ([3436c9a](https://github.com/kiharalab/Emap2lig/commit/3436c9a0c06af5fead2471958c7af4ed02a8602c))


### Bug Fixes

* **map:** pass --emdb-id through to contour level lookup ([7dfbc98](https://github.com/kiharalab/Emap2lig/commit/7dfbc9866c78797347bdf3b9b2de85b5f16bb3c6))
* **web:** make the cache directory setting reach model loading ([ad21666](https://github.com/kiharalab/Emap2lig/commit/ad21666c4009e3a72d5117bc14800face5667d18))


### Documentation

* **readme:** serve DeepWiki badge from shields.io ([f3073d7](https://github.com/kiharalab/Emap2lig/commit/f3073d73c5c55507c73cda8f1b8953320d0e7d28))

## [0.4.4](https://github.com/kiharalab/Emap2lig/compare/v0.4.3...v0.4.4) (2026-08-14)


### Bug Fixes

* **build:** stop Lightning from retaining predict outputs in host memory ([f21bce0](https://github.com/kiharalab/Emap2lig/commit/f21bce048779404c0c3ae7ffdfb1eddcfaede39b))
* **build:** stop Lightning from retaining predict outputs in host memory ([e3db398](https://github.com/kiharalab/Emap2lig/commit/e3db3987e37e25c9398a650ad02c5fbb337c5a65))

## [0.4.3](https://github.com/kiharalab/Emap2lig/compare/v0.4.2...v0.4.3) (2026-06-23)


### Bug Fixes

* **build:** cap multiplicity memory and document GPU settings ([091b7d0](https://github.com/kiharalab/Emap2lig/commit/091b7d0c9149f1b6b6c99eaecef19b67fed3d493))
* **build:** keep multiplicity chunking within one predict pass ([999f78b](https://github.com/kiharalab/Emap2lig/commit/999f78bd0f366ca4ba6dc61497dd76eac022f203))
* **build:** keep multiplicity chunking within one predict pass ([fa3eddf](https://github.com/kiharalab/Emap2lig/commit/fa3eddfc4174c7f3c118e90de2bf7c0b8477d542))

## Changelog

All notable changes to this project will be documented in this file.

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
Update this file by hand when cutting a release.
