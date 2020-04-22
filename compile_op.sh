# 1. manually edit CUDA_DIR and TENSORFLOW_LIB in config.sh
# ./pointnet_plusplus/utils/tf_ops/config.sh

# 2. complie
cd ./pointnet_plusplus/utils/tf_ops
cd grouping && bash tf_grouping_compile.sh
cd ../sampling && bash tf_sampling_compile.sh
cd ../3d_interpolation/ && bash tf_interpolate_compile.sh
echo 'tf_ops building done!'