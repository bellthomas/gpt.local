for f in $(find ~/.cache/huggingface/datasets -name 'cache-*.arrow'); do rm $f; done
for f in $(find ~/.cache/huggingface/datasets -name 'tmp*'); do rm $f; done