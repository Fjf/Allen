# Allen CI configuration

## Adding new devices

1. Add an entry for the device to `devices.yaml`. Set `TARGET`, `DEVICE_ID`, and the `tag:` accordingly

2. Add a job entry to run in the minimal pipeline: e.g.
```yaml
epyc7502:
  extends:
    - .epyc7502
    - .run_job

```

3. Add a job entry to run in the full pipeline, taking care to `extends:` from the right key based on the `TARGET` of the device:
```yaml
epyc7502-full:
  extends:
    - .epyc7502
    - .[cuda/hip/cpu]_run_job
    - .run_jobs_full
```

4. If you added a new CUDA device - did you check `OVERRIDE_CUDA_ARCH_FLAG` in `.gitlab-ci.yml` contains the right flags for this device?  

## Adding new tests

See [Gitlab CI documentation](https://docs.gitlab.com/ee/ci/yaml/#parallel-matrix-jobs) for more information on how the `parallel:matrix` keyword works.

### to the minimal pipeline

Add a key to `.run_matrix_jobs_minimal:parallel:matrix:` in `common-run.yaml` e.g.
```yaml
      # efficiency tests
      - TEST_NAME: "run_physics_efficiency" # name of the test - runs the bash script scripts/ci/jobs/$TEST_NAME.sh
        SEQUENCE: ["hlt1_pp_validation"]    # sequence(s) to run the test on
        INPUT_FILES: ["bsphiphi_mag_down_201907"] # input dataset
```

Other variables can be set (but are optional - see below).

### to the full pipeline

Add a key to `.run_matrix_jobs_full:parallel:matrix:` in `common-run.yaml` e.g.
```yaml
      - TEST_NAME: "run_throughput"     # name of the test - runs the bash script scripts/ci/jobs/$TEST_NAME.sh
        BUILD_TYPE: ["RelWithDebInfo"]  # use RelWithDebInfo build
#        OPTIONS: [""]                  # leave out for default build, with no additional build options
        SEQUENCE: ["hlt1_pp_default"]   # sequence
        INPUT_FILES: ["SMOG2_pppHe"]    # dataset name
#        GEOMETRY: [""]                 # don't add this, to use the default geom
```

If your test needs a build of Allen that is not yet included in the `build` stage, you will need to create one.

In order to ensure the correct build from the `build` stage is used in your test, make sure that the

- `${LCG_ARCHITECTURE}` (default value is set by `.run_jobs` key)
- `${BUILD_TYPE}` (default is `RelWithDebInfo` set by `.run_jobs` key)
- `${SEQUENCE}` (must be set in `.run_matrix_jobs_full:parallel:matrix:`)
- `${OPTIONS}` (optional, can be set in `.run_matrix_jobs_full:parallel:matrix:`)
- `${GEOMETRY}` (optional, can be left undefined or set if a specific geometry is needed)

variables are correctly set and match.

## Adding new efficiency reference files

Create the reference file with the format `test/reference/${DATA_TAG}_${DEVICE_ID}.txt`.


## Adding new builds

The `parallel:matrix:` keys will need to be modified in either `.build_job_minimal_matrix` or `.build_job_additional_matrix`.

N.B. 

- `$TARGET` does not need to be set in `parallel:matrix:` for the full builds, but it will need to be for the minimal builds.
- `$OPTIONS` can be left blank or undefined. If options need to be passed to CMake e.g. `-DBUILD_TESTS=ON -DENABLE_CONTRACTS=ON`, then `$OPTIONS` can be set to `BUILD_TESTS+ENABLE_CONTRACTS` which will set both CMake options to `ON` by default. If you need this to be something other than `ON`, then you can do `BUILD_TESTS=OFF+ENABLE_CONTRACTS=OFF`, for example.
- In downstream `run`-stage jobs, the `$OPTIONS` variable content *must* match for the build to be found properly.