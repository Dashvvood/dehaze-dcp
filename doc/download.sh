#！/bin/bash

name_list=(
    CVPR09_HE.pdf
    ECCV2010_HE.pdf
    TPAMI2011_HE.pdf
    TPAMI2013_HE.pdf
    CVPR09_slides.pdf
    ECCV10_slides.pdf
    EURASIP2016_LEE.pdf
)

url_list=(
    https://kaiminghe.github.io/publications/cvpr09.pdf
    https://kaiminghe.github.io/publications/eccv10guidedfilter.pdf
    https://kaiminghe.github.io/publications/pami10dehaze.pdf
    https://kaiminghe.github.io/publications/pami12guidedfilter.pdf
    https://kaiminghe.github.io/cvpr09/cvpr09slides.pdf
    https://kaiminghe.github.io/cvpr09/eccv10ppt.pdf
    https://jivp-eurasipjournals.springeropen.com/counter/pdf/10.1186/s13640-016-0104-y.pdf
)

for ((i=0; i<${#name_list[@]}; i++)); do
    if [ -f "${name_list[$i]}" ]; then
        echo "already exist: ${name_list[$i]}"
    else
        wget -O ${name_list[$i]} ${url_list[$i]}
    fi
done
