src_dir_1=/export/c11/sli136/l2mt/en-es/data/raw
src_key_1=all-mixed
src_dir_2=/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors
src_key_2=para
dest_dir=/export/c11/sli136/l2mt/en-es/data/raw
dest_key=all-mixed-para

[ ! -d ${dest_dir}/${dest_key} ] && mkdir -p ${dest_dir}/${dest_key}

dest_file_dev=${dest_dir}/${dest_key}/dev.${dest_key}
dest_file_test=${dest_dir}/${dest_key}/test.${dest_key}
dest_file_train=${dest_dir}/${dest_key}/train.${dest_key}

# cat ${src_dir_1}/${src_key_1}/dev.${src_key_1}.en > ${dest_file_dev}.en
# cat ${src_dir_1}/${src_key_1}/dev.${src_key_1}.es > ${dest_file_dev}.es

# cat ${src_dir_1}/${src_key_1}/test.${src_key_1}.en > ${dest_file_test}.en
# cat ${src_dir_1}/${src_key_1}/test.${src_key_1}.es > ${dest_file_test}.es

# cat ${src_dir_1}/${src_key_1}/train.${src_key_1}.en > ${dest_file_train}.en
# cat ${src_dir_1}/${src_key_1}/train.${src_key_1}.es > ${dest_file_train}.es

cp ${src_dir_1}/${src_key_1}/dev.${src_key_1}.en ${dest_file_dev}.en
cp ${src_dir_1}/${src_key_1}/dev.${src_key_1}.es ${dest_file_dev}.es
echo "done copying grammar-mixed to dev"

cp ${src_dir_1}/${src_key_1}/test.${src_key_1}.en ${dest_file_test}.en
cp ${src_dir_1}/${src_key_1}/test.${src_key_1}.es ${dest_file_test}.es
echo "done copying grammar-mixed to test"

# cp ${src_dir_1}/${src_key_1}/train.${src_key_1}.en ${dest_file_train}.en
# cp ${src_dir_1}/${src_key_1}/train.${src_key_1}.es ${dest_file_train}.es
# echo "done copying grammar-mixed to train"


cat ${src_dir_2}/dev.${src_key_2}.en >> ${dest_file_dev}.en
cat ${src_dir_2}/dev.${src_key_2}.es >> ${dest_file_dev}.es
echo "done copying para to dev"

cat ${src_dir_2}/test.${src_key_2}.en >> ${dest_file_test}.en
cat ${src_dir_2}/test.${src_key_2}.es >> ${dest_file_test}.es
echo "done copying para to test"

cat ${src_dir_2}/train.${src_key_2}.en >> ${dest_file_train}.en
cat ${src_dir_2}/train.${src_key_2}.es >> ${dest_file_train}.es
echo "done copying para to train"

cat ${src_dir_2}/dev.${src_key_2}.en >> ${dest_file_dev}.en
cat ${src_dir_2}/dev.${src_key_2}.es >> ${dest_file_dev}.es
echo "done copying para to dev"

cat ${src_dir_2}/test.${src_key_2}.en >> ${dest_file_test}.en
cat ${src_dir_2}/test.${src_key_2}.es >> ${dest_file_test}.es
echo "done copying para to test"

cat ${src_dir_2}/train.${src_key_2}.en >> ${dest_file_train}.en
cat ${src_dir_2}/train.${src_key_2}.es >> ${dest_file_train}.es
echo "done copying para to train"
