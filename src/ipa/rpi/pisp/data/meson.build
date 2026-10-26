# SPDX-License-Identifier: CC0-1.0

conf_files = files([
    'imx219.json',
    'imx219_noir.json',
    'imx283.json',
    'imx290.json',
    'imx296.json',
    'imx296_mono.json',
    'imx335.json',
    'imx355.json',
    'imx378.json',
    'imx415.json',
    'imx415_b0569.json',
    'imx462.json',
    'imx477.json',
    'imx477_noir.json',
    'imx477_scientific.json',
    'imx500.json',
    'imx519.json',
    'imx662.json',
    'imx708.json',
    'imx708_noir.json',
    'imx708_wide.json',
    'imx708_wide_noir.json',
    'mira220.json',
    'ov5647.json',
    'ov5647_noir.json',
    'ov64a40.json',
    'ov9281_mono.json',
    'se327m12.json',
    'uncalibrated.json',
    'vd55g1.json',
    'vd55g1_mono.json',
    'vd56g3.json',
    'vd56g3_mono.json',
])

install_data(conf_files,
             install_dir : ipa_data_dir / 'rpi' / 'pisp')

# Models for rpi.denoise_bstm, which finds them by file name.
tflite_files = files([
    'tflite/imx662_32_2e13w_16x274x484_g32-192_c025_d100_int8.tflite',
    'tflite/imx662_32_2e13w_16x60x60_g32-192_c025_d100_fp32.tflite',
    'tflite/imx662_32_2e13w_16x80x80_g32-192_c025_d100_int8.tflite',
    'tflite/imx662_64_2e12_16x274x484_g32-192_c025_d100_int8.tflite',
    'tflite/imx662_64_2e12_16x80x80_g32-192_c025_d100_int8.tflite',
])

install_data(tflite_files,
             install_dir : ipa_data_dir / 'rpi' / 'pisp' / 'tflite')
