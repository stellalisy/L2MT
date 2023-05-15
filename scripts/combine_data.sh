error='runon'

clean_dir='/home/sli136/l2mt/data/raw/'
artl2_dir='/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors/'
mixed_dir='/home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors/'
out_dir=/export/c11/sli136/l2mt/data/raw/artl2-${error}/

[ ! -d ${out_dir} ] && mkdir -p ${out_dir}

splits='dev,test,train'

sr_name_clean='clean'
sr_name_art=${error}
tg_name=mixed-${error}

for split in ${splits//,/ }; do
    clean_file=${clean_dir}/${split}.${sr_name_clean}.en
    artl2_file=${artl2_dir}/${split}.${sr_name_art}.en
    out_file=${out_dir}/${split}.${tg_name}.en
    cat ${clean_file} >> ${out_file}
    cat ${artl2_file} >> ${out_file}

    clean_file=${clean_dir}/${split}.${sr_name_clean}.es
    artl2_file=${artl2_dir}/${split}.${sr_name_art}.es
    out_file=${out_dir}/${split}.${tg_name}.es
    cat ${clean_file} >> ${out_file}
    cat ${artl2_file} >> ${out_file}
done