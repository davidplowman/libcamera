# SPDX-License-Identifier: CC0-1.0

rpi_ipa_controller_sources = files([
    'algorithm.cpp',
    'controller.cpp',
    'device_status.cpp',
    'histogram.cpp',
    'rpi/af.cpp',
    'rpi/agc.cpp',
    'rpi/agc_channel.cpp',
    'rpi/alsc.cpp',
    'rpi/awb.cpp',
    'rpi/awb_bayes.cpp',
    'rpi/black_level.cpp',
    'rpi/cac.cpp',
    'rpi/ccm.cpp',
    'rpi/contrast.cpp',
    'rpi/decompand.cpp',
    'rpi/denoise.cpp',
    'rpi/denoise_ncnn.cpp',
    'rpi/dpc.cpp',
    'rpi/geq.cpp',
    'rpi/global_motion.cpp',
    'rpi/hdr.cpp',
    'rpi/lux.cpp',
    'rpi/model_denoise.cpp',
    'rpi/noise.cpp',
    'rpi/saturation.cpp',
    'rpi/sdn.cpp',
    'rpi/sharpen.cpp',
    'rpi/sync.cpp',
    'rpi/temporal_accum.cpp',
    'rpi/tonemap.cpp',
])

rpi_ipa_controller_deps = [
    libcamera_private,
]

tflite_dep = dependency('tensorflow-lite', required : get_option('rpi-awb-nn'))

if tflite_dep.found()
    rpi_ipa_controller_sources += files([
        'rpi/awb_nn.cpp',
    ])
    rpi_ipa_controller_deps += tflite_dep
endif

ncnn_dep = dependency('ncnn', required : true).as_system()
rpi_ipa_controller_deps += ncnn_dep

# BSTM (BrainStorm) NPU offload of the model_denoise predictor trunk.
# The SDK ships no pkg-config file and installs nowhere standard, so the
# tree is located by the bstm_root option rather than probed for. Absent or
# incomplete, model_denoise still builds and runs everything on the CPU --
# RPI_HAVE_BSTM simply stays undefined.
bstm_root = get_option('bstm_root')
if bstm_root != ''
    bstm_inc = bstm_root / 'core' / 'include'
    # The 'target' flavour is the one with client mode compiled in, which
    # is what lets us share the NPU with the running bstorm daemon.
    bstm_lib = cxx.find_library('bstorm_core',
                                dirs : [bstm_root / 'core' / 'out' / 'core.7712c0.2.64.target'],
                                required : false)
    if bstm_lib.found() and cxx.has_header('bstorm_init.h', args : ['-I' + bstm_inc])
        rpi_ipa_controller_deps += declare_dependency(
            dependencies : bstm_lib,
            include_directories : include_directories(bstm_inc, is_system : true),
            compile_args : ['-DRPI_HAVE_BSTM'])
        summary({'BSTM NPU offload' : bstm_root}, section : 'Raspberry Pi IPA')
    else
        warning('bstm_root set but libbstorm_core / core/include not found under ' + bstm_root)
    endif
endif

# ncnn is built with NCNN_OPENMP, so its static library pulls in undefined
# GOMP_parallel/omp_* references that ncnn.pc does not declare a dependency
# on. Enabling OpenMP here both resolves those symbols and lets our own
# #pragma omp blocks (model_denoise.cpp and friends) actually thread.
openmp_dep = dependency('openmp', required : true)
rpi_ipa_controller_deps += openmp_dep

rpi_ipa_controller_lib = static_library('rpi_ipa_controller', rpi_ipa_controller_sources,
                                        include_directories : libipa_includes,
                                        dependencies : rpi_ipa_controller_deps)
